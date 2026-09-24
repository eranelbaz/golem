//! Substrate-free GGUF v3 low-level reader shared by golem-engine and xtask:
//! type tags, bounded chunked reads, string decoding, and header prologue.
//! Higher-level metadata parsing lives with each consumer.

use std::fmt;
use std::io::Read;

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
}
