//! Minimal GGUF v3 writer (little-endian, 32-byte tensor alignment).
//!
//! Scope: models only the metadata value types the tiny-moe generator writes
//! (U32/F32/Str scalars and I32/F32/Str arrays) — not a general GGUF serializer.

use std::io::Write;

/// ggml tensor-type id used in the tensor table. Distinct namespace from the
/// GGUF metadata value-type tags below: `GGML_TYPE_F32 = 0`, `GGUF_TYPE_F32 = 6`.
pub const GGML_TYPE_F32: u32 = 0;
// GGUF metadata value-type tags (ggml/src/gguf.cpp).
const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_I32: u32 = 5;
const GGUF_TYPE_F32: u32 = 6;
const GGUF_TYPE_STR: u32 = 8;
const GGUF_TYPE_ARR: u32 = 9;
const ALIGN: u64 = 32;

/// Metadata value types the generator writes (all owned — no lifetimes).
pub enum GgufValue {
    U32(u32),
    F32(f32),
    Str(String),
    ArrI32(Vec<i32>),
    ArrF32(Vec<f32>),
    ArrStr(Vec<String>),
}

impl GgufValue {
    fn tag(&self) -> u32 {
        match self {
            GgufValue::U32(_) => GGUF_TYPE_U32,
            GgufValue::F32(_) => GGUF_TYPE_F32,
            GgufValue::Str(_) => GGUF_TYPE_STR,
            GgufValue::ArrI32(_) | GgufValue::ArrF32(_) | GgufValue::ArrStr(_) => GGUF_TYPE_ARR,
        }
    }
}

pub struct GgufKv {
    pub key: String,
    pub value: GgufValue,
}

/// Tensor-table entry; `dims` is in ggml order (ne[0] innermost).
pub struct GgufTensorMeta {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_kv(out: &mut Vec<u8>, kv: &GgufKv) {
    // key first, then type tag — the order ggml/src/gguf.cpp's reader expects
    put_str(out, &kv.key);
    out.extend_from_slice(&kv.value.tag().to_le_bytes());
    match &kv.value {
        GgufValue::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Str(s) => put_str(out, s),
        GgufValue::ArrI32(v) => {
            out.extend_from_slice(&GGUF_TYPE_I32.to_le_bytes());
            out.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        GgufValue::ArrF32(v) => {
            out.extend_from_slice(&GGUF_TYPE_F32.to_le_bytes());
            out.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        GgufValue::ArrStr(v) => {
            out.extend_from_slice(&GGUF_TYPE_STR.to_le_bytes());
            out.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for s in v {
                put_str(out, s);
            }
        }
    }
}

/// Write a complete GGUF v3 file: 24-byte header, KV section (key first, then
/// type tag), tensor table with data-section-relative cumulative padded offsets
/// starting at 0, an alignment pad, then blob+pad per tensor.
///
/// Fails with `ErrorKind::InvalidData` if a tensor's `ggml_type` is not
/// `GGML_TYPE_F32` or its payload is not exactly `dims.product() * 4` bytes.
pub fn write_gguf(
    w: &mut impl Write,
    kvs: &[GgufKv],
    tensors: &[(GgufTensorMeta, Vec<u8>)],
) -> std::io::Result<()> {
    let mut head = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    head.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for kv in kvs {
        put_kv(&mut head, kv);
    }

    // Tensor table: offsets are DATA-SECTION-RELATIVE, cumulative, padded
    // (ggml/src/gguf.cpp starts ctx->offset at the padded table end and
    // validates each ti.offset against that running counter).
    let mut table = Vec::new();
    let mut offset = 0u64;
    for (t, data) in tensors {
        let name = &t.name;
        if t.ggml_type != GGML_TYPE_F32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{name}: only GGML_TYPE_F32 tensors are supported"),
            ));
        }
        let expected = t.dims.iter().product::<u64>() as usize * 4;
        let got = data.len();
        if got != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{name}: payload {got} bytes, expected {expected}"),
            ));
        }
        put_str(&mut table, &t.name);
        table.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            table.extend_from_slice(&d.to_le_bytes());
        }
        table.extend_from_slice(&t.ggml_type.to_le_bytes());
        table.extend_from_slice(&offset.to_le_bytes());
        offset = align_up(offset + data.len() as u64);
    }

    w.write_all(&head)?;
    w.write_all(&table)?;
    // pad between table and first blob (ggml/src/gguf.cpp seeks to GGML_PAD(tell, ALIGN))
    let table_end = (head.len() + table.len()) as u64;
    let head_pad = align_up(table_end) - table_end;
    w.write_all(&[0u8; 32][..head_pad as usize])?;
    let mut pos = table_end + head_pad;
    for (_, data) in tensors {
        w.write_all(data)?;
        pos += data.len() as u64;
        let blob_pad = align_up(pos) - pos;
        pos += blob_pad;
        w.write_all(&[0u8; 32][..blob_pad as usize])?;
    }
    w.flush()
}

fn align_up(v: u64) -> u64 {
    v.div_ceil(ALIGN) * ALIGN
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv_bytes() -> Vec<u8> {
        let kvs = vec![GgufKv {
            key: "general.architecture".into(),
            value: GgufValue::Str("glm4moe".into()),
        }];
        let tensors = vec![(
            GgufTensorMeta {
                name: "output_norm.weight".into(),
                dims: vec![4],
                ggml_type: GGML_TYPE_F32,
            },
            [1.0f32; 4].map(|f| f.to_le_bytes()).concat(),
        )];
        let mut buf = Vec::new();
        write_gguf(&mut buf, &kvs, &tensors).unwrap();
        buf
    }

    #[test]
    fn magic_version_counts() {
        let b = kv_bytes();
        assert_eq!(&b[0..4], b"GGUF");
        assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), 3);
        assert_eq!(u64::from_le_bytes(b[8..16].try_into().unwrap()), 1); // tensors
        assert_eq!(u64::from_le_bytes(b[16..24].try_into().unwrap()), 1); // kvs
    }

    #[test]
    fn kv_record_is_key_then_type() {
        // The pinned loader reads key first, then type (ggml/src/gguf.cpp,
        // gguf_init_from_reader): a KV record starts with the u64 key length.
        let b = kv_bytes();
        let mut i = 24;
        let klen = u64::from_le_bytes(b[i..i + 8].try_into().unwrap()) as usize;
        i += 8;
        assert_eq!(&b[i..i + klen], b"general.architecture");
        i += klen;
        assert_eq!(u32::from_le_bytes(b[i..i + 4].try_into().unwrap()), 8); // kv type STR
        i += 4;
        let vlen = u64::from_le_bytes(b[i..i + 8].try_into().unwrap()) as usize;
        i += 8;
        assert_eq!(&b[i..i + vlen], b"glm4moe");
    }

    #[test]
    fn tensor_data_starts_32_byte_aligned() {
        // The writer pads between the table and the first blob; anchor at the
        // computed table end (head + KV section + table, rounded up to 32)
        // instead of a payload needle scan.
        let b = kv_bytes();
        let mut i = 24usize; // end of the fixed 24-byte header

        let klen = u64::from_le_bytes(b[i..i + 8].try_into().unwrap()) as usize;
        i += 8 + klen + 4; // key + STR tag
        let vlen = u64::from_le_bytes(b[i..i + 8].try_into().unwrap()) as usize;
        i += 8 + vlen;
        let nlen = u64::from_le_bytes(b[i..i + 8].try_into().unwrap()) as usize;
        i += 8 + nlen;
        let ndims = u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
        i += 4 + ndims * 8 + 4 + 8; // dims + ggml_type + offset
        let data_start = align_up(i as u64);
        assert_eq!(data_start % 32, 0, "first blob start not 32-aligned");
        assert!(
            b[i..data_start as usize].iter().all(|&x| x == 0),
            "gap before first blob must be pad"
        );
        assert_eq!(
            &b[data_start as usize..data_start as usize + 4],
            &1.0f32.to_le_bytes()[..]
        );
        assert_eq!(b.len() as u64, data_start + 32);
    }

    #[test]
    fn arr_kv_types_round_trip_layout() {
        let kvs = [
            GgufKv {
                key: "tokenizer.ggml.token_type".into(),
                value: GgufValue::ArrI32(vec![1, 2]),
            },
            GgufKv {
                key: "tokenizer.ggml.scores".into(),
                value: GgufValue::ArrF32(vec![0.0, 0.0]),
            },
        ];
        let mut buf = Vec::new();
        write_gguf(&mut buf, &kvs, &[]).unwrap();
        let b = &buf[..];
        // first record: keylen(8) key arrtype(4) elemtype(4) nelem(8)
        let keylen = u64::from_le_bytes(b[24..32].try_into().unwrap()) as usize;
        let mut i = 32 + keylen;
        assert_eq!(u32::from_le_bytes(b[i..i + 4].try_into().unwrap()), 9); // ARRAY
        i += 4;
        assert_eq!(u32::from_le_bytes(b[i..i + 4].try_into().unwrap()), 5); // I32
        i += 4;
        assert_eq!(u64::from_le_bytes(b[i..i + 8].try_into().unwrap()), 2); // n = 2
    }

    #[test]
    fn every_blob_start_is_32_byte_aligned() {
        // Multi-tensor: EVERY blob start (data_start + declared table offset)
        // must be 32-aligned. Offsets come from the parsed table, not a scan.
        let mk = |name: &str, n: u64| {
            let meta = GgufTensorMeta {
                name: name.into(),
                dims: vec![n],
                ggml_type: GGML_TYPE_F32,
            };
            (meta, vec![0xABu8; n as usize * 4])
        };
        let tensors = [mk("a.weight", 3), mk("b.weight", 40), mk("c.weight", 1)];
        let mut buf = Vec::new();
        write_gguf(&mut buf, &[], &tensors).unwrap();
        let b = &buf[..];
        assert_eq!(u64::from_le_bytes(b[8..16].try_into().unwrap()), 3); // tensors
        assert_eq!(u64::from_le_bytes(b[16..24].try_into().unwrap()), 0); // kvs

        // walk the tensor table (starts at byte 24 — no KVs, head is 24 bytes)
        let mut i = 24usize;
        let mut starts = Vec::new(); // (declared offset, payload bytes) per tensor
        for _ in 0..3 {
            let klen = u64::from_le_bytes(b[i..i + 8].try_into().unwrap()) as usize;
            i += 8 + klen;
            let ndims = u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
            i += 4;
            let mut ne = 1u64;
            for _ in 0..ndims {
                ne *= u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
                i += 8;
            }
            i += 4; // ggml_type
            let off = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
            i += 8;
            starts.push((off, ne * 4));
        }
        let data_start = align_up(i as u64);
        let mut end = data_start;
        for (k, (off, len)) in starts.iter().enumerate() {
            let start = data_start + off;
            assert_eq!(start % 32, 0, "tensor {k} blob start not 32-aligned");
            assert_eq!(&b[start as usize..start as usize + 4], &[0xABu8; 4][..]);
            end = end.max(start + len);
        }
        assert_eq!(b.len() as u64, align_up(end));
    }

    #[test]
    fn rejects_bad_tensor_payload() {
        let meta = |ty: u32| GgufTensorMeta {
            name: "x.weight".into(),
            dims: vec![4],
            ggml_type: ty,
        };
        let mut buf = Vec::new();
        let short = [(meta(GGML_TYPE_F32), vec![0u8; 12])]; // 4 dims => 16 bytes
        let err = write_gguf(&mut buf, &[], &short).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let not_f32 = [(meta(1), vec![0u8; 16])];
        let err = write_gguf(&mut buf, &[], &not_f32).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
