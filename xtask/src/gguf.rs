//! Minimal GGUF v3 writer (little-endian, 32-byte tensor alignment).
//!
//! Scope: models only the metadata value types the tiny-moe generator writes
//! (U32/F32/Str scalars and I32/F32/Str arrays) — not a general GGUF serializer.

use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;

/// ggml tensor-type id — distinct namespace from the GGUF value-type tags in golem-gguf.
pub const GGML_TYPE_F32: u32 = 0;
// GGUF v3 value-type tags written by this generator.
const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_I32: u32 = 5;
const GGUF_TYPE_F32: u32 = 6;
const GGUF_TYPE_STR: u32 = 8;
const GGUF_TYPE_ARR: u32 = 9;
const ALIGN: u64 = 32;

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

/// Returns the byte offset of the tensor-data section (aligned table end).
/// Fails with `InvalidData` if `ggml_type != GGML_TYPE_F32` or a payload is not
/// exactly `dims.product() * 4` bytes.
pub fn write_gguf(
    w: &mut impl Write,
    kvs: &[GgufKv],
    tensors: &[(GgufTensorMeta, Vec<u8>)],
) -> std::io::Result<u64> {
    let mut head = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    head.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for kv in kvs {
        put_kv(&mut head, kv);
    }

    let mut table = Vec::new();
    let mut offset = 0u64;
    for (t, data) in tensors {
        let name = &t.name;
        // F32 rows are ne-product × 4; quant rows come from the shared block
        // table. Anything without a block layout is refused.
        let expected = if t.ggml_type == GGML_TYPE_F32 {
            t.dims.iter().product::<u64>() as usize * 4
        } else {
            let (ne0, rest) = t.dims.split_first().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{name}: rank-0 tensor"),
                )
            })?;
            let row = golem_gguf::ggml_row_bytes(t.ggml_type, *ne0).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{name}: ggml type {} has no block layout", t.ggml_type),
                )
            })?;
            (row * rest.iter().product::<u64>()) as usize
        };
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
    let table_end = (head.len() + table.len()) as u64;
    let head_pad = align_up(table_end) - table_end;
    let data_start = table_end + head_pad;
    w.write_all(&[0u8; 32][..head_pad as usize])?;
    let mut pos = data_start;
    for (_, data) in tensors {
        w.write_all(data)?;
        pos += data.len() as u64;
        let blob_pad = align_up(pos) - pos;
        pos += blob_pad;
        w.write_all(&[0u8; 32][..blob_pad as usize])?;
    }
    w.flush()?;
    Ok(data_start)
}

fn align_up(v: u64) -> u64 {
    v.div_ceil(ALIGN) * ALIGN
}

/// Reads the `general.architecture` string KV from a GGUF v3 file without
/// touching tensor data. Errors (as `String`) on bad magic, unsupported
/// version, truncation, an unknown type tag, or a missing/non-string
/// architecture KV. Low-level decoding is shared with golem-engine via the
/// substrate-free `golem-gguf` crate.
pub fn read_arch(path: &Path) -> Result<String, String> {
    use golem_gguf as gg;

    const ARCH_KEY: &str = "general.architecture";
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut r = BufReader::new(file);

    let (_, _n_tensors, n_kv) = gg::read_header(&mut r).map_err(|e| e.0)?;

    for _ in 0..n_kv {
        let key = gg::read_string(&mut r).map_err(|e| e.0)?;
        let tag = gg::read_u32(&mut r).map_err(|e| e.0)?;
        if key == ARCH_KEY {
            if tag != gg::TYPE_STR {
                return Err(format!(
                    "{ARCH_KEY}: expected string value (type tag 8), got {tag}"
                ));
            }
            return gg::read_string(&mut r).map_err(|e| e.0);
        }
        gg::skip_value(&mut r, tag).map_err(|e| e.0)?;
    }
    Err(format!("missing {ARCH_KEY} KV"))
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
        let b = kv_bytes();
        let mut i = 24usize;

        let klen = u64::from_le_bytes(b[i..i + 8].try_into().unwrap()) as usize;
        i += 8 + klen + 4;
        let vlen = u64::from_le_bytes(b[i..i + 8].try_into().unwrap()) as usize;
        i += 8 + vlen;
        let nlen = u64::from_le_bytes(b[i..i + 8].try_into().unwrap()) as usize;
        i += 8 + nlen;
        let ndims = u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
        i += 4 + ndims * 8 + 4 + 8;
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

        let mut i = 24usize;
        let mut starts = Vec::new();
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
            i += 4;
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
        let short = [(meta(GGML_TYPE_F32), vec![0u8; 12])];
        let err = write_gguf(&mut buf, &[], &short).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let not_f32 = [(meta(1), vec![0u8; 16])];
        let err = write_gguf(&mut buf, &[], &not_f32).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    fn write_temp_model(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("golem-xtask-gguf-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("m.gguf");
        std::fs::write(&p, bytes).unwrap();
        p
    }

    fn arch_only_bytes() -> Vec<u8> {
        let kvs = vec![GgufKv {
            key: "general.architecture".into(),
            value: GgufValue::Str("glm4moe".into()),
        }];
        let mut buf = Vec::new();
        write_gguf(&mut buf, &kvs, &[]).unwrap();
        buf
    }

    fn cleanup(p: &std::path::Path) {
        let _ = std::fs::remove_dir_all(p.parent().unwrap());
    }

    #[test]
    fn read_arch_finds_arch_after_other_kvs_and_arrays() {
        let kvs = vec![
            GgufKv {
                key: "general.name".into(),
                value: GgufValue::Str("tiny".into()),
            },
            GgufKv {
                key: "tokenizer.ggml.token_type".into(),
                value: GgufValue::ArrI32(vec![1, 2, 3]),
            },
            GgufKv {
                key: "general.architecture".into(),
                value: GgufValue::Str("glm4moe".into()),
            },
        ];
        let mut buf = Vec::new();
        write_gguf(&mut buf, &kvs, &[]).unwrap();
        let p = write_temp_model("happy", &buf);
        assert_eq!(read_arch(&p).unwrap(), "glm4moe");
        cleanup(&p);
    }

    #[test]
    fn read_arch_rejects_bad_magic() {
        let p = write_temp_model("badmagic", b"NOPE");
        let err = read_arch(&p).unwrap_err();
        assert!(err.contains("bad magic"), "{err}");
        cleanup(&p);
    }

    #[test]
    fn read_arch_rejects_wrong_version() {
        let mut b = arch_only_bytes();
        b[4..8].copy_from_slice(&2u32.to_le_bytes());
        let p = write_temp_model("badver", &b);
        let err = read_arch(&p).unwrap_err();
        assert!(err.contains("version 2"), "{err}");
        cleanup(&p);
    }

    #[test]
    fn read_arch_missing_arch_kv_is_err() {
        let kvs = vec![GgufKv {
            key: "general.name".into(),
            value: GgufValue::Str("tiny".into()),
        }];
        let mut buf = Vec::new();
        write_gguf(&mut buf, &kvs, &[]).unwrap();
        let p = write_temp_model("noarch", &buf);
        let err = read_arch(&p).unwrap_err();
        assert!(err.contains("missing general.architecture"), "{err}");
        cleanup(&p);
    }

    #[test]
    fn read_arch_rejects_non_string_arch() {
        let kvs = vec![GgufKv {
            key: "general.architecture".into(),
            value: GgufValue::U32(7),
        }];
        let mut buf = Vec::new();
        write_gguf(&mut buf, &kvs, &[]).unwrap();
        let p = write_temp_model("archtype", &buf);
        let err = read_arch(&p).unwrap_err();
        assert!(err.contains("expected string"), "{err}");
        cleanup(&p);
    }

    #[test]
    fn read_arch_rejects_truncated_file() {
        let b = arch_only_bytes();
        let p = write_temp_model("trunc", &b[..b.len() / 2]);
        let err = read_arch(&p).unwrap_err();
        assert!(err.contains("truncated"), "{err}");
        cleanup(&p);
    }
}
