//! Substrate-free GGUF v3 low-level reader shared by golem-engine and xtask:
//! type tags, bounded chunked reads, string decoding, and header prologue.
//! Higher-level metadata parsing lives with each consumer.

use std::fmt;
use std::io::{Read, Seek};

/// GGUF v3 value-type tags (ggml/include/gguf.h): 9 = array, 12 = f64.
pub const TYPE_U8: u32 = 0;
/// Signed 8-bit integer tag.
pub const TYPE_I8: u32 = 1;
/// Unsigned 16-bit integer tag.
pub const TYPE_U16: u32 = 2;
/// Signed 16-bit integer tag.
pub const TYPE_I16: u32 = 3;
/// Unsigned 32-bit integer tag.
pub const TYPE_U32: u32 = 4;
/// Signed 32-bit integer tag.
pub const TYPE_I32: u32 = 5;
/// 32-bit float tag.
pub const TYPE_F32: u32 = 6;
/// Boolean (one byte) tag.
pub const TYPE_BOOL: u32 = 7;
/// UTF-8 string tag.
pub const TYPE_STR: u32 = 8;
/// Homogeneous array tag.
pub const TYPE_ARRAY: u32 = 9;
/// Unsigned 64-bit integer tag.
pub const TYPE_U64: u32 = 10;
/// Signed 64-bit integer tag.
pub const TYPE_I64: u32 = 11;
/// 64-bit float tag.
pub const TYPE_F64: u32 = 12;

/// GGUF file magic, `b"GGUF"`.
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// Upper bound per `read_exact` call in [`read_bytes`], so a bogus length
/// cannot drive a huge single allocation before the short read is detected.
const READ_CHUNK: usize = 64 * 1024;

/// A GGUF parse failure, message-only.
#[derive(Debug, Clone)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

fn short_read(e: std::io::Error) -> Error {
    Error(format!("truncated GGUF metadata: {e}"))
}

/// Reads one little-endian byte.
pub fn read_u8(r: &mut impl Read) -> Result<u8, Error> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).map_err(short_read)?;
    Ok(b[0])
}

/// Reads one little-endian `u16`.
pub fn read_u16(r: &mut impl Read) -> Result<u16, Error> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(short_read)?;
    Ok(u16::from_le_bytes(b))
}

/// Reads one little-endian `u32`.
pub fn read_u32(r: &mut impl Read) -> Result<u32, Error> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(short_read)?;
    Ok(u32::from_le_bytes(b))
}

/// Reads one little-endian `u64`.
pub fn read_u64(r: &mut impl Read) -> Result<u64, Error> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(short_read)?;
    Ok(u64::from_le_bytes(b))
}

/// Reads `len` bytes in bounded chunks; allocation grows only as bytes are
/// actually read, so a forged length fails at EOF before consuming memory.
pub fn read_bytes(r: &mut impl Read, len: u64) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    let mut remaining = len;
    while remaining > 0 {
        let chunk = remaining.min(READ_CHUNK as u64) as usize;
        let start = out.len();
        out.resize(start + chunk, 0);
        r.read_exact(&mut out[start..]).map_err(short_read)?;
        remaining -= chunk as u64;
    }
    Ok(out)
}

/// Reads and discards `len` bytes in bounded chunks.
pub fn discard(r: &mut impl Read, len: u64) -> Result<(), Error> {
    let mut buf = vec![0u8; READ_CHUNK];
    let mut remaining = len;
    while remaining > 0 {
        let chunk = remaining.min(READ_CHUNK as u64) as usize;
        r.read_exact(&mut buf[..chunk]).map_err(short_read)?;
        remaining -= chunk as u64;
    }
    Ok(())
}

/// GGUF string encoding: u64 length prefix + UTF-8 bytes.
pub fn read_string(r: &mut impl Read) -> Result<String, Error> {
    let len = read_u64(r)?;
    let bytes = read_bytes(r, len)?;
    String::from_utf8(bytes).map_err(|e| Error(format!("invalid UTF-8 in GGUF string: {e}")))
}

/// Consumes one KV value payload of type `tag` without materializing it.
/// Nested arrays are rejected, matching gguf.h — nothing writes them and
/// unbounded recursion on hostile input would overflow the stack.
pub fn skip_value(r: &mut impl Read, tag: u32) -> Result<(), Error> {
    match tag {
        TYPE_U8 | TYPE_I8 | TYPE_BOOL => discard(r, 1),
        TYPE_U16 | TYPE_I16 => discard(r, 2),
        TYPE_U32 | TYPE_I32 | TYPE_F32 => discard(r, 4),
        TYPE_U64 | TYPE_I64 | TYPE_F64 => discard(r, 8),
        TYPE_STR => {
            let len = read_u64(r)?;
            discard(r, len)
        }
        TYPE_ARRAY => {
            let elem_tag = read_u32(r)?;
            if elem_tag == TYPE_ARRAY {
                return Err(Error("nested GGUF arrays are not supported".into()));
            }
            let n = read_u64(r)?;
            for _ in 0..n {
                skip_value(r, elem_tag)?;
            }
            Ok(())
        }
        other => Err(Error(format!("unknown GGUF value type tag {other}"))),
    }
}

/// Reads and validates the GGUF v3 header prologue: magic, version, and the
/// v3 field order (n_tensors precedes n_kv, per ggml/include/gguf.h).
pub fn read_header(r: &mut impl Read) -> Result<(u32, u64, u64), Error> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(short_read)?;
    if magic != GGUF_MAGIC {
        return Err(Error(format!(
            "bad magic {:?} (expected \"GGUF\")",
            String::from_utf8_lossy(&magic)
        )));
    }
    let version = read_u32(r)?;
    if version != 3 {
        return Err(Error(format!(
            "unsupported GGUF version {version} (expected 3)"
        )));
    }
    let n_tensors = read_u64(r)?;
    let n_kv = read_u64(r)?;
    Ok((version, n_tensors, n_kv))
}

/// GGML tensor-element type tags (ggml.h `GGML_TYPE_*`).
pub const GGML_TYPE_F32: u32 = 0;
/// 16-bit float.
pub const GGML_TYPE_F16: u32 = 1;
/// 4-bit quant, block 32.
pub const GGML_TYPE_Q4_0: u32 = 2;
/// 4-bit quant with minimum, block 32.
pub const GGML_TYPE_Q4_1: u32 = 3;
/// 5-bit quant, block 32.
pub const GGML_TYPE_Q5_0: u32 = 6;
/// 5-bit quant with minimum, block 32.
pub const GGML_TYPE_Q5_1: u32 = 7;
/// 8-bit quant, block 32.
pub const GGML_TYPE_Q8_0: u32 = 8;
/// 8-bit quant with minimum, block 32.
pub const GGML_TYPE_Q8_1: u32 = 9;
/// 2-bit K-quant, block 256.
pub const GGML_TYPE_Q2_K: u32 = 10;
/// 3-bit K-quant, block 256.
pub const GGML_TYPE_Q3_K: u32 = 11;
/// 4-bit K-quant, block 256.
pub const GGML_TYPE_Q4_K: u32 = 12;
/// 5-bit K-quant, block 256.
pub const GGML_TYPE_Q5_K: u32 = 13;
/// 6-bit K-quant, block 256.
pub const GGML_TYPE_Q6_K: u32 = 14;

/// (`elements_per_block`, `bytes_per_block`) for the types M2.3 reads. `None`
/// for types without a fixed block layout — the full table lands with M3.1's
/// store module.
#[must_use]
pub fn ggml_type_block(ggml_type: u32) -> Option<(u64, u64)> {
    let (blck, size) = match ggml_type {
        GGML_TYPE_F32 => (1, 4),
        GGML_TYPE_F16 => (1, 2),
        GGML_TYPE_Q4_0 => (32, 18),
        GGML_TYPE_Q4_1 => (32, 20),
        GGML_TYPE_Q5_0 => (32, 22),
        GGML_TYPE_Q5_1 => (32, 24),
        GGML_TYPE_Q8_0 => (32, 34),
        GGML_TYPE_Q8_1 => (32, 36),
        GGML_TYPE_Q2_K => (256, 84),
        GGML_TYPE_Q3_K => (256, 110),
        GGML_TYPE_Q4_K => (256, 144),
        GGML_TYPE_Q5_K => (256, 176),
        GGML_TYPE_Q6_K => (256, 210),
        _ => return None,
    };
    Some((blck, size))
}

/// Bytes of one packed row of `ne0` elements (partial leading block rounds up,
/// matching ggml row semantics). Quant strides are per-plane from this, never
/// per-element F32 sizes.
#[must_use]
pub fn ggml_row_bytes(ggml_type: u32, ne0: u64) -> Option<u64> {
    let (blck, size) = ggml_type_block(ggml_type)?;
    Some(ne0.div_ceil(blck) * size)
}

/// One tensor-table row (gguf.h `gguf_tensor_info` projection).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    /// Tensor name, e.g. `blk.1.ffn_gate_exps.weight`.
    pub name: String,
    /// Data offset RELATIVE to the aligned tensor-data section start
    /// ([`data_section_offset`]).
    pub offset: u64,
    /// First three dimensions; `0`-padded below rank 3.
    pub dims: [u64; 3],
    /// GGML element type tag ([`ggml_type_block`] interprets it).
    pub ggml_type: u32,
}

/// Walks the KV block, capturing `general.alignment` (default 32) and
/// rejecting non-U32 or non-power-of-two values (gguf.h requirement).
fn walk_kvs(r: &mut (impl Read + Seek), n_kv: u64) -> Result<u64, Error> {
    let mut alignment = 32u64;
    for _ in 0..n_kv {
        let key = read_string(r)?;
        let tag = read_u32(r)?;
        if key == "general.alignment" {
            if tag != TYPE_U32 {
                return Err(Error("general.alignment must be a U32".into()));
            }
            let a = u64::from(read_u32(r)?);
            if a == 0 || !a.is_power_of_two() {
                return Err(Error(format!("invalid general.alignment {a}")));
            }
            alignment = a;
        } else {
            skip_value(r, tag)?;
        }
    }
    Ok(alignment)
}

/// Reads the tensor-table rows following the KV block.
fn walk_table(r: &mut (impl Read + Seek), n_tensors: u64) -> Result<Vec<TensorEntry>, Error> {
    let mut out = Vec::new();
    for _ in 0..n_tensors {
        let name = read_string(r)?;
        let n_dims = read_u32(r)?;
        if n_dims == 0 || n_dims > 3 {
            return Err(Error(format!(
                "tensor {name}: unsupported rank {n_dims} (expected 1..=3)"
            )));
        }
        let mut dims = [0u64; 3];
        for d in dims.iter_mut().take(n_dims as usize) {
            *d = read_u64(r)?;
        }
        let ggml_type = read_u32(r)?;
        let offset = read_u64(r)?;
        out.push(TensorEntry {
            name,
            offset,
            dims,
            ggml_type,
        });
    }
    Ok(out)
}

/// Absolute file offset of the tensor-data section: end of the tensor table
/// (which follows the KV block) aligned up to `general.alignment` — the layout
/// gguf.h and the fixture writer produce. Rewinds to the start first.
///
/// # Errors
/// GGUF parse failures (bad magic/version, truncated KV or tensor table).
pub fn data_section_offset(r: &mut (impl Read + Seek)) -> Result<u64, Error> {
    r.rewind().map_err(short_read)?;
    let (_, n_tensors, n_kv) = read_header(r)?;
    let alignment = walk_kvs(r, n_kv)?;
    walk_table(r, n_tensors)?;
    let pos = r.stream_position().map_err(short_read)?;
    Ok(pos.div_ceil(alignment) * alignment)
}

/// Walks the tensor table. Entry [`TensorEntry::offset`]s are relative to the
/// aligned section start ([`data_section_offset`]). Rewinds to the start
/// first.
///
/// # Errors
/// GGUF parse failures, or a tensor with rank outside 1..=3 (M2.3's readers
/// are 3-dim; the general walker lands with M3.1's store).
pub fn tensor_entries(r: &mut (impl Read + Seek)) -> Result<Vec<TensorEntry>, Error> {
    r.rewind().map_err(short_read)?;
    let (_, n_tensors, n_kv) = read_header(r)?;
    walk_kvs(r, n_kv)?;
    walk_table(r, n_tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skip_value_rejects_nested_arrays() {
        // array<u32, [array<u32, [7]>]>: outer tag 9, elem tag 9, count 1,
        // then the inner array (elem tag 4, count 1, value 7).
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        assert_eq!(
            skip_value(&mut &bytes[..], TYPE_ARRAY).unwrap_err().0,
            "nested GGUF arrays are not supported"
        );
    }

    #[test]
    fn read_header_rejects_bad_magic_and_wrong_version() {
        assert!(read_header(&mut &b"NOPE"[..]).is_err());
        let mut v2 = Vec::new();
        v2.extend_from_slice(b"GGUF");
        v2.extend_from_slice(&2u32.to_le_bytes());
        assert!(read_header(&mut &v2[..]).is_err());
    }

    /// Minimal GGUF v3 bytes: `alignment` kv, one str kv, one 2-dim F32 tensor.
    fn sample_gguf(alignment: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        b.extend_from_slice(&2u64.to_le_bytes()); // n_kv
        for (k, tag, v) in [
            (
                "general.alignment",
                TYPE_U32,
                alignment.to_le_bytes().to_vec(),
            ),
            ("general.name", TYPE_STR, {
                let mut v = 2u64.to_le_bytes().to_vec();
                v.extend_from_slice(b"hi");
                v
            }),
        ] {
            b.extend_from_slice(k.len().to_le_bytes().as_slice());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&v);
        }
        b.extend_from_slice(4u64.to_le_bytes().as_slice()); // name len
        b.extend_from_slice(b"w.gt");
        b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&32u64.to_le_bytes()); // ne0
        b.extend_from_slice(&16u64.to_le_bytes()); // ne1
        b.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
        b.extend_from_slice(&64u64.to_le_bytes()); // offset (aligned by writer)
        b
    }

    #[test]
    fn tensor_walk_and_alignment() {
        let b = sample_gguf(64);
        let mut cur = std::io::Cursor::new(&b);
        let dso = data_section_offset(&mut cur).unwrap();
        assert_eq!(dso % 64, 0, "section start aligned to general.alignment");
        assert!(dso >= b.len() as u64 && dso < b.len() as u64 + 64);
        let entries = tensor_entries(&mut cur).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0],
            TensorEntry {
                name: "w.gt".into(),
                offset: 64,
                dims: [32, 16, 0],
                ggml_type: GGML_TYPE_F32,
            }
        );
    }

    #[test]
    fn default_alignment_is_32_and_rows_round_up_blocks() {
        let b = sample_gguf(32);
        let mut cur = std::io::Cursor::new(&b);
        let d0 = data_section_offset(&mut cur).unwrap();
        assert_eq!(d0 % 32, 0, "default alignment 32 applies without the kv");
        assert_eq!(ggml_row_bytes(GGML_TYPE_F32, 32), Some(128));
        assert_eq!(ggml_row_bytes(GGML_TYPE_Q8_0, 32), Some(34));
        assert_eq!(
            ggml_row_bytes(GGML_TYPE_Q4_0, 30),
            Some(18),
            "partial block rounds up"
        );
        assert_eq!(ggml_row_bytes(99, 32), None, "unknown type has no layout");
        // Quant plane stride is per-plane, not per-element F32:
        assert_eq!(ggml_row_bytes(GGML_TYPE_Q8_0, 32).unwrap() * 16, 544);
        assert_ne!(
            ggml_row_bytes(GGML_TYPE_Q8_0, 32).unwrap() * 16,
            32 * 16 * 4
        );
    }

    #[test]
    fn tensor_walk_rejects_rank_4() {
        let mut b = sample_gguf(32);
        // Drop the tensor row (name-len prefix .. offset) and re-append it as 4-dim.
        let row_len = 8 + 4 + 4 + 8 + 8 + 4 + 8; // len + "w.gt" + n_dims + 2 dims + type + offset
        b.truncate(b.len() - row_len);
        b.extend_from_slice(4u64.to_le_bytes().as_slice());
        b.extend_from_slice(b"w.gt");
        b.extend_from_slice(&4u32.to_le_bytes());
        for _ in 0..4 {
            b.extend_from_slice(&8u64.to_le_bytes());
        }
        b.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        let mut cur = std::io::Cursor::new(&b);
        assert!(tensor_entries(&mut cur).is_err());
    }

    #[test]
    fn bad_alignment_is_rejected() {
        let b = sample_gguf(0);
        let mut cur = std::io::Cursor::new(&b);
        assert!(data_section_offset(&mut cur).is_err());
    }
}
