//! Minimal GGUF v3 metadata reader (little-endian, pure Rust).
//!
//! Reads only the header, KV records, and tensor infos; never maps tensor
//! data. Byte-size and alignment handling are deferred to their first
//! consumers (VRAM budget).

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use golem_gguf as raw_gguf;
use golem_gguf::TYPE_ARRAY;

use crate::error::EngineError;

impl From<raw_gguf::Error> for EngineError {
    fn from(e: raw_gguf::Error) -> Self {
        EngineError::InvalidModel(e.to_string())
    }
}

use raw_gguf::{read_header, read_string, read_u8, read_u16, read_u32, read_u64};

/// Reads a value payload for an already-consumed type tag.
fn read_payload(r: &mut impl Read, tag: u32) -> Result<GgufValue, EngineError> {
    match tag {
        raw_gguf::TYPE_U8 => Ok(GgufValue::U8(read_u8(r)?)),
        raw_gguf::TYPE_I8 => Ok(GgufValue::I8(read_u8(r)? as i8)),
        raw_gguf::TYPE_U16 => Ok(GgufValue::U16(read_u16(r)?)),
        raw_gguf::TYPE_I16 => Ok(GgufValue::I16(read_u16(r)? as i16)),
        raw_gguf::TYPE_U32 => Ok(GgufValue::U32(read_u32(r)?)),
        raw_gguf::TYPE_I32 => Ok(GgufValue::I32(read_u32(r)? as i32)),
        raw_gguf::TYPE_F32 => Ok(GgufValue::F32(f32::from_bits(read_u32(r)?))),
        raw_gguf::TYPE_BOOL => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        raw_gguf::TYPE_STR => Ok(GgufValue::Str(read_string(r)?)),
        // Array: element type u32, element count u64, then the elements.
        // Nested arrays are rejected, matching gguf.h — nothing writes them and
        // unbounded recursion on hostile input would overflow the stack.
        TYPE_ARRAY => {
            let elem_tag = read_u32(r)?;
            if elem_tag == TYPE_ARRAY {
                return Err(EngineError::InvalidModel(
                    "nested GGUF arrays are not supported".into(),
                ));
            }
            let n = read_u64(r)?;
            let mut elems = Vec::new();
            for _ in 0..n {
                elems.push(read_payload(r, elem_tag)?);
            }
            Ok(GgufValue::Array(elems))
        }
        raw_gguf::TYPE_U64 => Ok(GgufValue::U64(read_u64(r)?)),
        raw_gguf::TYPE_I64 => Ok(GgufValue::I64(read_u64(r)? as i64)),
        raw_gguf::TYPE_F64 => Ok(GgufValue::F64(f64::from_bits(read_u64(r)?))),
        other => Err(EngineError::InvalidModel(format!(
            "unknown GGUF value type tag {other}"
        ))),
    }
}

fn read_value(r: &mut impl Read) -> Result<GgufValue, EngineError> {
    let tag = read_u32(r)?;
    read_payload(r, tag)
}

fn read_tensor_meta(r: &mut impl Read) -> Result<TensorMeta, EngineError> {
    let name = read_string(r)?;
    let n_dims = read_u32(r)?;
    if n_dims > 4 {
        return Err(EngineError::InvalidModel(format!(
            "{}: n_dims {n_dims} exceeds the 4-dim GGUF tensor shape limit",
            name
        )));
    }
    let mut ne = [0u64; 4];
    for slot in ne.iter_mut().take(n_dims as usize) {
        *slot = read_u64(r)?;
    }
    let ggml_type = read_u32(r)?;
    let offset = read_u64(r)?;
    Ok(TensorMeta {
        name,
        n_dims,
        ne,
        ggml_type,
        offset,
    })
}

/// A parsed GGUF v3 header: version, KV metadata records, and tensor infos.
pub struct GgufMetadata {
    /// GGUF version (only 3 is accepted).
    pub version: u32,
    /// KV metadata records in file order.
    pub kvs: Vec<(String, GgufValue)>,
    /// Tensor infos in file order.
    pub tensors: Vec<TensorMeta>,
}

impl GgufMetadata {
    /// Returns the string value of the KV record with this key, if any.
    pub fn kv_str(&self, key: &str) -> Option<&str> {
        self.kvs
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| v.as_str())
    }

    /// Returns the unsigned integer value of the KV record with this key,
    /// widening u8/u16/u32 to u64, if any.
    pub fn kv_u64(&self, key: &str) -> Option<u64> {
        self.kvs
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| v.as_u64())
    }
}

/// A GGUF metadata value, tagged per the v3 type table.
#[derive(Clone)]
pub enum GgufValue {
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Signed 8-bit integer.
    I8(i8),
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Signed 16-bit integer.
    I16(i16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Signed 32-bit integer.
    I32(i32),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 64-bit integer.
    I64(i64),
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
    /// Boolean (one byte, nonzero = true).
    Bool(bool),
    /// UTF-8 string.
    Str(String),
    /// Homogeneous array of values.
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Returns the string slice if this value is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the value widened to u64 if this is an unsigned integer type.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(u64::from(*v)),
            GgufValue::U16(v) => Some(u64::from(*v)),
            GgufValue::U32(v) => Some(u64::from(*v)),
            GgufValue::U64(v) => Some(*v),
            _ => None,
        }
    }
}

/// A GGUF tensor info: shape and dtype only; no tensor data is read.
pub struct TensorMeta {
    /// Tensor name.
    pub name: String,
    /// Number of dimensions present in `ne` (rest of `ne` is zero-filled).
    pub n_dims: u32,
    /// Dimension extents, innermost first; zero-padded to 4 entries.
    pub ne: [u64; 4],
    /// ggml tensor-type id.
    pub ggml_type: u32,
    /// Offset of the tensor data from the start of the tensor-data section.
    pub offset: u64,
}

/// Reads GGUF v3 metadata (header, KVs, tensor infos) from `path`.
///
/// Never maps tensor data — tensor `offset`s are recorded, not followed.
pub fn read_metadata(path: &Path) -> Result<GgufMetadata, EngineError> {
    let file = File::open(path)?;
    let mut r = BufReader::new(file);

    let (version, n_tensors, n_kv) = read_header(&mut r)?;

    let mut kvs = Vec::new();
    for _ in 0..n_kv {
        let key = read_string(&mut r)?;
        let value = read_value(&mut r)?;
        kvs.push((key, value));
    }

    let mut tensors = Vec::new();
    for _ in 0..n_tensors {
        tensors.push(read_tensor_meta(&mut r)?);
    }

    Ok(GgufMetadata {
        version,
        kvs,
        tensors,
    })
}

/// Shared in-memory metadata builder for engine tests.
#[cfg(test)]
pub(crate) mod test_gguf {
    use super::*;

    /// Tensor descriptor for building in-memory metadata in tests.
    pub(crate) struct Tensor {
        /// Tensor name.
        pub(crate) name: String,
        /// Number of dimensions present in `ne`.
        pub(crate) n_dims: u32,
        /// Dimension extents, zero-padded to 4 entries.
        pub(crate) ne: [u64; 4],
        /// ggml tensor-type id.
        pub(crate) ggml_type: u32,
    }

    /// Builds a `GgufMetadata` directly — no file round-trip; `sanity::check`
    /// consumes the struct.
    pub(crate) fn metadata_with(kvs: &[(String, GgufValue)], tensors: &[Tensor]) -> GgufMetadata {
        GgufMetadata {
            version: 3,
            kvs: kvs.to_vec(),
            tensors: tensors
                .iter()
                .map(|t| TensorMeta {
                    name: t.name.clone(),
                    n_dims: t.n_dims,
                    ne: t.ne,
                    ggml_type: t.ggml_type,
                    offset: 0,
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use xtask::gguf::{GgufKv, GgufTensorMeta, GgufValue as XValue, write_gguf};

    fn sample_gguf() -> Vec<u8> {
        let kvs = [
            GgufKv {
                key: "general.architecture".into(),
                value: XValue::Str("glm4moe".into()),
            },
            GgufKv {
                key: "tiny_moe.test".into(),
                value: XValue::U32(42),
            },
        ];
        let tensors = [(
            GgufTensorMeta {
                name: "blk.0.attn_q.weight".into(),
                dims: vec![2, 3, 1],
                ggml_type: 0,
            },
            vec![0u8; 24],
        )];
        let mut b = Vec::new();
        write_gguf(&mut b, &kvs, &tensors).unwrap();
        b
    }

    #[test]
    fn reads_kv_types_and_tensor_meta() {
        let dir = std::env::temp_dir().join("golem-engine-gguf-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("sample.gguf");
        std::fs::write(&p, sample_gguf()).unwrap();

        let m = read_metadata(&p).unwrap();
        assert_eq!(m.version, 3);
        assert_eq!(m.kv_str("general.architecture"), Some("glm4moe"));
        assert_eq!(m.kv_u64("tiny_moe.test"), Some(42));
        assert_eq!(m.tensors.len(), 1);
        let t = &m.tensors[0];
        assert_eq!(t.name, "blk.0.attn_q.weight");
        assert_eq!(t.n_dims, 3);
        assert_eq!(&t.ne[..3], &[2, 3, 1]);
    }

    #[test]
    fn rejects_nested_arrays() {
        // array<u32, [array<u32, [7]>]>: outer tag 9, elem tag 9, count 1,
        // then the inner array (elem tag 4, count 1, value 7).
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        assert!(matches!(
            read_value(&mut &bytes[..]),
            Err(EngineError::InvalidModel(_))
        ));
    }

    #[test]
    fn rejects_bad_magic_and_truncation() {
        let dir = std::env::temp_dir().join("golem-engine-gguf-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("bad.gguf");
        std::fs::write(&p, b"NOPE").unwrap();
        assert!(matches!(
            read_metadata(&p),
            Err(EngineError::InvalidModel(_))
        ));

        let p2 = dir.join("trunc.gguf");
        std::fs::write(&p2, &sample_gguf()[..40]).unwrap();
        assert!(matches!(
            read_metadata(&p2),
            Err(EngineError::InvalidModel(_))
        ));
    }
}
