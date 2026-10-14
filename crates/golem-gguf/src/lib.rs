//! Substrate-free GGUF v3 low-level reader shared by golem-engine and xtask:
//! type tags, bounded chunked reads, string decoding, and header prologue.
//! Higher-level metadata parsing lives with each consumer.

use std::collections::HashSet;
use std::io::{self, Read, Seek};
use std::path::{Path, PathBuf};

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

/// A GGUF parse failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Underlying I/O failure other than a short read.
    #[error("GGUF I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The stream ended inside a header, KV, or tensor-table field.
    #[error("truncated GGUF metadata")]
    UnexpectedEof,
    /// The first four bytes are not [`GGUF_MAGIC`].
    #[error("bad magic {0:?} (expected \"GGUF\")")]
    BadMagic(String),
    /// Only GGUF version 3 is supported.
    #[error("unsupported GGUF version {0} (expected 3)")]
    Version(u32),
    /// Structurally invalid content (bad tag, rank, alignment, UTF-8, ...).
    #[error("{0}")]
    Invalid(String),
}

fn short_read(e: std::io::Error) -> Error {
    if e.kind() == std::io::ErrorKind::UnexpectedEof {
        Error::UnexpectedEof
    } else {
        Error::Io(e)
    }
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
    String::from_utf8(bytes)
        .map_err(|e| Error::Invalid(format!("invalid UTF-8 in GGUF string: {e}")))
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
                return Err(Error::Invalid(
                    "nested GGUF arrays are not supported".into(),
                ));
            }
            let n = read_u64(r)?;
            for _ in 0..n {
                skip_value(r, elem_tag)?;
            }
            Ok(())
        }
        other => Err(Error::Invalid(format!(
            "unknown GGUF value type tag {other}"
        ))),
    }
}

/// Reads and validates the GGUF v3 header prologue: magic, version, and the
/// v3 field order (n_tensors precedes n_kv, per ggml/include/gguf.h).
pub fn read_header(r: &mut impl Read) -> Result<(u32, u64, u64), Error> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(short_read)?;
    if magic != GGUF_MAGIC {
        return Err(Error::BadMagic(
            String::from_utf8_lossy(&magic).into_owned(),
        ));
    }
    let version = read_u32(r)?;
    if version != 3 {
        return Err(Error::Version(version));
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
/// 2.06-bit i-quant, block 256.
pub const GGML_TYPE_IQ2_XXS: u32 = 16;
/// 2.31-bit i-quant, block 256.
pub const GGML_TYPE_IQ2_XS: u32 = 17;
/// 3.06-bit i-quant, block 256.
pub const GGML_TYPE_IQ3_XXS: u32 = 18;
/// 1.56-bit i-quant, block 256.
pub const GGML_TYPE_IQ1_S: u32 = 19;
/// 4-bit non-linear i-quant, block 32.
pub const GGML_TYPE_IQ4_NL: u32 = 20;
/// 3.44-bit i-quant, block 256.
pub const GGML_TYPE_IQ3_S: u32 = 21;
/// 2.5-bit i-quant, block 256.
pub const GGML_TYPE_IQ2_S: u32 = 22;
/// 4.25-bit i-quant, block 256.
pub const GGML_TYPE_IQ4_XS: u32 = 23;
/// 1.75-bit i-quant, block 256.
pub const GGML_TYPE_IQ1_M: u32 = 29;
/// bfloat16.
pub const GGML_TYPE_BF16: u32 = 30;

/// (`elements_per_block`, `bytes_per_block`) for the types the pool reads.
/// `None` for types without a fixed block layout.
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
        GGML_TYPE_IQ2_XXS => (256, 66),
        GGML_TYPE_IQ2_XS => (256, 74),
        GGML_TYPE_IQ3_XXS => (256, 98),
        GGML_TYPE_IQ1_S => (256, 50),
        GGML_TYPE_IQ4_NL => (32, 18),
        GGML_TYPE_IQ3_S => (256, 110),
        GGML_TYPE_IQ2_S => (256, 82),
        GGML_TYPE_IQ4_XS => (256, 136),
        GGML_TYPE_IQ1_M => (256, 56),
        GGML_TYPE_BF16 => (1, 2),
        _ => return None,
    };
    Some((blck, size))
}

/// Bytes of one packed row of `ne0` elements. `None` for an unknown type or
/// when `ne0` is not a whole number of blocks (ggml rejects such rows).
/// Quant strides are per-plane from this, never per-element F32 sizes.
#[must_use]
pub fn ggml_row_bytes(ggml_type: u32, ne0: u64) -> Option<u64> {
    let (blck, size) = ggml_type_block(ggml_type)?;
    if !ne0.is_multiple_of(blck) {
        return None;
    }
    (ne0 / blck).checked_mul(size)
}

/// One tensor-table row (gguf.h `gguf_tensor_info` projection).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    /// Tensor name, e.g. `blk.1.ffn_gate_exps.weight`.
    pub name: String,
    /// Data offset RELATIVE to the aligned tensor-data section start
    /// ([`data_section_offset`]).
    pub offset: u64,
    /// Rank as stored in the file (1..=3).
    pub n_dims: u32,
    /// First three dimensions; `0`-padded below rank 3.
    pub dims: [u64; 3],
    /// GGML element type tag ([`ggml_type_block`] interprets it).
    pub ggml_type: u32,
}

impl TensorEntry {
    /// Expert plane kind, or `None` for non-expert names and for expert-named
    /// tensors whose rank is not 3 (same rule as the C `golem_tensor_is_expert`).
    #[must_use]
    pub fn expert_kind(&self) -> Option<ExpertKind> {
        if self.n_dims == 3 {
            ExpertKind::from_tensor_name(&self.name)
        } else {
            None
        }
    }
}

/// Maximum tensor-name length in bytes, `GGML_MAX_NAME` minus the NUL.
const MAX_TENSOR_NAME: usize = 63;

/// Routed-expert plane kind. One matcher for C and Rust semantics
/// (§6.6): the exact stem `ffn_{gate,up,down}_exps` at a name-segment
/// boundary, optionally followed by `.weight`. Everything else — the router
/// `ffn_gate_inp`, dense `ffn_gate`, and loader-created `.scale` variants — is
/// not an expert plane. Fused `ffn_gate_up_exps` is excluded (P2 splits
/// routed gate/up/down ranges independently).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertKind {
    /// `ffn_gate_exps`.
    Gate,
    /// `ffn_up_exps`.
    Up,
    /// `ffn_down_exps`.
    Down,
}

impl ExpertKind {
    /// All kinds, in ordinal order.
    pub const ALL: [ExpertKind; 3] = [Self::Gate, Self::Up, Self::Down];

    /// Parses from a ggml or GGUF tensor name. Names of `GGML_MAX_NAME` (64)
    /// bytes or more cannot exist in ggml and never match.
    pub fn from_tensor_name(name: &str) -> Option<Self> {
        if name.len() > MAX_TENSOR_NAME {
            return None;
        }
        let base = name.strip_suffix(".weight").unwrap_or(name);
        for kind in Self::ALL {
            if let Some(prefix) = base.strip_suffix(kind.stem())
                && (prefix.is_empty() || prefix.ends_with('.'))
            {
                return Some(kind);
            }
        }
        None
    }

    /// Tensor-name stem (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`).
    pub fn stem(self) -> &'static str {
        match self {
            Self::Gate => "ffn_gate_exps",
            Self::Up => "ffn_up_exps",
            Self::Down => "ffn_down_exps",
        }
    }

    /// Flat ordinal (Gate=0, Up=1, Down=2).
    pub fn ordinal(self) -> usize {
        self as usize
    }
}

/// Walks the KV block once: integer/string values are returned, other tags
/// are skipped. `general.alignment` (default 32) must be a power-of-two U32
/// (gguf.h). Duplicate keys are rejected, as gguf.h does.
fn walk_kvs(r: &mut (impl Read + Seek), n_kv: u64) -> Result<(u64, Vec<(String, KvVal)>), Error> {
    let mut alignment = 32u64;
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for _ in 0..n_kv {
        let key = read_string(r)?;
        if !seen.insert(key.clone()) {
            return Err(Error::Invalid(format!("duplicate GGUF key {key}")));
        }
        let tag = read_u32(r)?;
        if key == "general.alignment" {
            if tag != TYPE_U32 {
                return Err(Error::Invalid("general.alignment must be a U32".into()));
            }
            let a = u64::from(read_u32(r)?);
            if a == 0 || !a.is_power_of_two() {
                return Err(Error::Invalid(format!("invalid general.alignment {a}")));
            }
            alignment = a;
            out.push((key, KvVal::Int(a)));
            continue;
        }
        if let Some(v) = read_int(r, tag)? {
            out.push((key, KvVal::Int(v)));
            continue;
        }
        let val = match tag {
            TYPE_STR => KvVal::Str(read_string(r)?),
            // Vocab-sized arrays (~151k token types on the target) are
            // never needed; key order is unspecified, so match by prefix.
            TYPE_ARRAY if key.starts_with("tokenizer.") => {
                skip_value(r, tag)?;
                continue;
            }
            TYPE_ARRAY => match read_int_array(r)? {
                Some(v) => KvVal::IntArr(v),
                None => continue,
            },
            _ => {
                skip_value(r, tag)?;
                continue;
            }
        };
        out.push((key, val));
    }
    Ok((alignment, out))
}

/// Reads one integer-like value of type `tag`, widened; `None` (nothing
/// consumed) for non-integer tags.
fn read_int(r: &mut impl Read, tag: u32) -> Result<Option<u64>, Error> {
    Ok(Some(match tag {
        TYPE_U8 => u64::from(read_u8(r)?),
        TYPE_I8 => read_u8(r)? as i8 as u64,
        TYPE_U16 => u64::from(read_u16(r)?),
        TYPE_I16 => read_u16(r)? as i16 as i64 as u64,
        TYPE_U32 => u64::from(read_u32(r)?),
        TYPE_I32 => read_u32(r)? as i32 as i64 as u64,
        TYPE_U64 => read_u64(r)?,
        TYPE_I64 => read_u64(r)? as i64 as u64,
        TYPE_BOOL => u64::from(read_u8(r)? != 0),
        _ => return Ok(None),
    }))
}

/// Reads an array payload (after its `TYPE_ARRAY` tag): integer elements
/// widened like scalars; any other element type is skipped (`None`).
fn read_int_array(r: &mut impl Read) -> Result<Option<Vec<u64>>, Error> {
    let elem = read_u32(r)?;
    if elem == TYPE_ARRAY {
        return Err(Error::Invalid(
            "nested GGUF arrays are not supported".into(),
        ));
    }
    let n = read_u64(r)?;
    let mut out = Vec::new();
    for _ in 0..n {
        match read_int(r, elem)? {
            Some(v) => out.push(v),
            None => skip_value(r, elem)?,
        }
    }
    Ok((out.len() as u64 == n).then_some(out))
}

/// Looks up an arch-prefixed integer-or-array KV, expanded to `n` per-layer
/// values (a scalar applies to every layer, as llama.cpp's `get_key_or_arr`).
pub fn arch_kv_per_layer(kvs: &[(String, KvVal)], suffix: &str, n: usize) -> Option<Vec<u64>> {
    let arch = kv_str(kvs, "general.architecture")?;
    let key = format!("{arch}.{suffix}");
    match &kvs.iter().find(|(k, _)| *k == key)?.1 {
        KvVal::Int(v) => Some(vec![*v; n]),
        KvVal::IntArr(v) if v.len() >= n => Some(v[..n].to_vec()),
        _ => None,
    }
}

/// Reads the tensor-table rows following the KV block.
fn walk_table(r: &mut (impl Read + Seek), n_tensors: u64) -> Result<Vec<TensorEntry>, Error> {
    let mut out = Vec::new();
    for _ in 0..n_tensors {
        let name = read_string(r)?;
        let n_dims = read_u32(r)?;
        if n_dims == 0 || n_dims > 3 {
            return Err(Error::Invalid(format!(
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
            n_dims,
            dims,
            ggml_type,
        });
    }
    Ok(out)
}

/// The KV values the placement/store readers consume; anything else is
/// skipped, never parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvVal {
    /// Any integer tag, widened.
    Int(u64),
    /// A string value.
    Str(String),
    /// An integer array, widened (per-layer hparams such as
    /// `attention.head_count_kv`).
    IntArr(Vec<u64>),
}

/// Reads the KV block as `(key, value)` pairs for the integer/string values
/// the Rust readers need; all other tags are skipped. Rewinds to the start
/// first.
///
/// # Errors
/// GGUF parse failures (bad magic/version, truncated KV block).
pub fn read_kvs(r: &mut (impl Read + Seek)) -> Result<Vec<(String, KvVal)>, Error> {
    r.rewind().map_err(short_read)?;
    let (_, _, n_kv) = read_header(r)?;
    Ok(walk_kvs(r, n_kv)?.1)
}

/// Looks up an integer KV by exact key.
#[must_use]
pub fn kv_int(kvs: &[(String, KvVal)], key: &str) -> Option<u64> {
    kvs.iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| match v {
            KvVal::Int(i) => Some(*i),
            _ => None,
        })
}

/// Looks up a string KV by exact key.
#[must_use]
pub fn kv_str<'a>(kvs: &'a [(String, KvVal)], key: &str) -> Option<&'a str> {
    kvs.iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| match v {
            KvVal::Str(s) => Some(s.as_str()),
            _ => None,
        })
}

/// Looks up an arch-prefixed integer KV (`<arch>.<suffix>`) by suffix.
#[must_use]
pub fn arch_kv_int(kvs: &[(String, KvVal)], suffix: &str) -> Option<u64> {
    let arch = kv_str(kvs, "general.architecture")?;
    kv_int(kvs, &format!("{arch}.{suffix}"))
}

/// Live expert blocks (§2.2, R-2): `[leading_dense, block_count − nextn)`.
/// MTP/NextN blocks excluded; an empty or inverted range is a corrupt header.
pub fn live_layers(kvs: &[(String, KvVal)]) -> Result<std::ops::Range<u32>, Error> {
    let get = |k: &str| arch_kv_int(kvs, k);
    let bc = get("block_count").ok_or_else(|| Error::Invalid("missing block_count".into()))?;
    let nextn = get("nextn_predict_layers").unwrap_or(0);
    let lead = get("leading_dense_block_count").unwrap_or(0);
    let end = bc.checked_sub(nextn).ok_or_else(|| {
        Error::Invalid(format!("nextn_predict_layers {nextn} > block_count {bc}"))
    })?;
    if lead >= end {
        return Err(Error::Invalid(format!(
            "no live layers: lead {lead}, block_count {bc}, nextn {nextn}"
        )));
    }
    let to32 = |v: u64| {
        u32::try_from(v).map_err(|_| Error::Invalid(format!("block index {v} out of range")))
    };
    Ok(to32(lead)?..to32(end)?)
}

/// `blk.<N>.` prefix parse.
#[must_use]
pub fn blk_of(name: &str) -> Option<u32> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// Sibling-shard path for shard `no` of `count`, rewriting the standard
/// `-NNNNN-of-MMMMM.gguf` suffix of `first`. Single-file models (`count ==
/// 1`) return `first` unchanged.
pub fn shard_path(first: &Path, no: u64, count: u64) -> std::io::Result<PathBuf> {
    if count == 1 {
        return Ok(first.to_path_buf());
    }
    let name = first.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("non-UTF-8 model path {first:?}"),
        )
    })?;
    let (stem, ext) = name.rsplit_once('.').unwrap_or((name, ""));
    let Some((prefix, counts)) = stem.rsplit_once("-of-") else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "split.count = {count} but {name} does not end in -NNNNN-of-MMMMM.{ext:?}; \
                 cannot discover sibling shards"
            ),
        ));
    };
    let Some((base, idx)) = prefix.rsplit_once('-') else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("cannot split shard index from {name}"),
        ));
    };
    if !idx.bytes().all(|b| b.is_ascii_digit()) || idx.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("shard index {idx:?} in {name} is not numeric"),
        ));
    }
    let width = idx.len();
    let ext_dot = if ext.is_empty() {
        String::new()
    } else {
        format!(".{ext}")
    };
    Ok(first.with_file_name(format!("{base}-{:0width$}-of-{counts}{ext_dot}", no + 1)))
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
    let (alignment, _) = walk_kvs(r, n_kv)?;
    walk_table(r, n_tensors)?;
    let pos = r.stream_position().map_err(short_read)?;
    Ok(pos.div_ceil(alignment) * alignment)
}

/// Walks the tensor table. Entry [`TensorEntry::offset`]s are relative to the
/// aligned section start ([`data_section_offset`]). Rewinds to the start
/// first.
///
/// # Errors
/// GGUF parse failures, or a tensor with rank outside 1..=3 (the table is
/// projected onto three dims).
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
    fn stem_round_trips_through_the_matcher() {
        for kind in ExpertKind::ALL {
            let name = format!("blk.3.{}.weight", kind.stem());
            assert_eq!(ExpertKind::from_tensor_name(&name), Some(kind));
        }
    }

    #[test]
    fn expert_name_matcher_exact_stem_and_weight() {
        assert_eq!(
            ExpertKind::from_tensor_name("blk.1.ffn_gate_exps.weight"),
            Some(ExpertKind::Gate)
        );
        assert_eq!(
            ExpertKind::from_tensor_name("blk.44.ffn_down_exps"),
            Some(ExpertKind::Down)
        );
        assert_eq!(
            ExpertKind::from_tensor_name("ffn_up_exps.weight"),
            Some(ExpertKind::Up)
        );
        // Loader .scale variants, router, dense FFN: never expert planes.
        assert_eq!(
            ExpertKind::from_tensor_name("blk.1.ffn_gate_exps.scale"),
            None
        );
        assert_eq!(
            ExpertKind::from_tensor_name("blk.1.ffn_gate_inp.weight"),
            None
        );
        assert_eq!(ExpertKind::from_tensor_name("blk.0.ffn_gate.weight"), None);
        assert_eq!(
            ExpertKind::from_tensor_name("blk.0.ffn_gate_up_exps.weight"),
            None
        );
    }

    #[test]
    fn int_arrays_widen_and_expand_per_layer() {
        // array<i32, [0, 1]>, then array<f32, [1.0]> (skipped).
        let mut b = TYPE_I32.to_le_bytes().to_vec();
        b.extend(2u64.to_le_bytes());
        b.extend(0i32.to_le_bytes());
        b.extend(1i32.to_le_bytes());
        assert_eq!(read_int_array(&mut &b[..]).unwrap(), Some(vec![0, 1]));
        let mut f = TYPE_F32.to_le_bytes().to_vec();
        f.extend(1u64.to_le_bytes());
        f.extend(1f32.to_le_bytes());
        assert_eq!(read_int_array(&mut &f[..]).unwrap(), None);

        let arch = ("general.architecture".to_owned(), KvVal::Str("a".into()));
        let kvs = [arch.clone(), ("a.h".into(), KvVal::IntArr(vec![0, 1, 2]))];
        assert_eq!(arch_kv_per_layer(&kvs, "h", 2), Some(vec![0, 1]));
        assert_eq!(arch_kv_per_layer(&kvs, "h", 4), None);
        let kvs = [arch, ("a.h".into(), KvVal::Int(7))];
        assert_eq!(arch_kv_per_layer(&kvs, "h", 2), Some(vec![7, 7]));
    }

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
            skip_value(&mut &bytes[..], TYPE_ARRAY)
                .unwrap_err()
                .to_string(),
            "nested GGUF arrays are not supported"
        );
    }

    #[test]
    fn read_header_rejects_bad_magic_and_wrong_version() {
        assert!(matches!(
            read_header(&mut &b"NOPE\x03\0\0\0"[..]),
            Err(Error::BadMagic(_))
        ));
        let mut v2 = Vec::new();
        v2.extend_from_slice(b"GGUF");
        v2.extend_from_slice(&2u32.to_le_bytes());
        assert!(matches!(read_header(&mut &v2[..]), Err(Error::Version(2))));
    }

    #[test]
    fn only_short_reads_map_to_truncation() {
        assert!(matches!(
            read_header(&mut &b"GGU"[..]),
            Err(Error::UnexpectedEof)
        ));
        struct Broken;
        impl Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("disk on fire"))
            }
        }
        assert!(matches!(read_u32(&mut Broken), Err(Error::Io(_))));
    }

    #[test]
    fn expert_name_matcher_bounds_and_suffix() {
        let long = format!("{}.ffn_gate_exps.weight", "b".repeat(43));
        assert_eq!(long.len(), 64);
        assert_eq!(
            ExpertKind::from_tensor_name(&long),
            None,
            "64 bytes: too long"
        );
        let ok = format!("{}.ffn_gate_exps.weight", "b".repeat(42));
        assert_eq!(ok.len(), 63);
        assert_eq!(ExpertKind::from_tensor_name(&ok), Some(ExpertKind::Gate));
        assert_eq!(ExpertKind::from_tensor_name("x_ffn_gate_exps"), None);
        assert_eq!(
            ExpertKind::from_tensor_name("blk.1.ffn_gate_exps.weight.weight"),
            None
        );
    }

    #[test]
    fn expert_kind_requires_rank_3() {
        let e = |n_dims| TensorEntry {
            name: "blk.1.ffn_up_exps.weight".into(),
            offset: 0,
            n_dims,
            dims: [32, 4, 0],
            ggml_type: GGML_TYPE_F32,
        };
        assert_eq!(e(3).expert_kind(), Some(ExpertKind::Up));
        assert_eq!(e(2).expert_kind(), None);
    }

    #[test]
    fn row_bytes_requires_whole_blocks() {
        assert_eq!(ggml_row_bytes(GGML_TYPE_Q4_0, 64), Some(36));
        assert_eq!(ggml_row_bytes(GGML_TYPE_Q4_0, 30), None);
        assert_eq!(ggml_row_bytes(GGML_TYPE_Q4_K, 257), None);
    }

    #[test]
    fn duplicate_kv_keys_are_rejected() {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        for _ in 0..2 {
            b.extend_from_slice(&1u64.to_le_bytes());
            b.extend_from_slice(b"k");
            b.extend_from_slice(&TYPE_U32.to_le_bytes());
            b.extend_from_slice(&7u32.to_le_bytes());
        }
        let err = read_kvs(&mut std::io::Cursor::new(&b)).unwrap_err();
        assert!(err.to_string().contains("duplicate"), "{err}");
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
                n_dims: 2,
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
            None,
            "partial block is not a valid row"
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

    fn glm_kvs(bc: u64, lead: u64, nextn: u64) -> Vec<(String, KvVal)> {
        vec![
            ("general.architecture".into(), KvVal::Str("glm5next".into())),
            ("glm5next.block_count".into(), KvVal::Int(bc)),
            (
                "glm5next.leading_dense_block_count".into(),
                KvVal::Int(lead),
            ),
            ("glm5next.nextn_predict_layers".into(), KvVal::Int(nextn)),
        ]
    }

    #[test]
    fn live_layers_excludes_dense_lead_and_mtp() {
        assert_eq!(
            live_layers(&glm_kvs(46, 3, 1)).unwrap(),
            3..45,
            "42 live, blk.45 MTP out"
        );
    }

    #[test]
    fn live_layers_rejects_corrupt_ranges() {
        for (bc, lead, nextn) in [(2, 0, 3), (46, 50, 1), (4, 0, 4)] {
            assert!(
                matches!(
                    live_layers(&glm_kvs(bc, lead, nextn)),
                    Err(Error::Invalid(_))
                ),
                "{bc} {lead} {nextn}"
            );
        }
    }

    #[test]
    fn blk_of_parses_prefix_only() {
        assert_eq!(blk_of("blk.45.ffn_up_exps.weight"), Some(45));
        assert_eq!(blk_of("output.weight"), None);
        assert_eq!(blk_of("blk.x.ffn"), None);
    }

    #[test]
    fn shard_path_rewrites_index_keeping_width() {
        use std::path::Path;
        let p = |s: &str, no, n| shard_path(Path::new(s), no, n).unwrap();
        assert_eq!(
            p("/d/M-00001-of-00006.gguf", 5, 6),
            Path::new("/d/M-00006-of-00006.gguf")
        );
        assert_eq!(
            p("/d/a-b-001-of-002.gguf", 1, 2),
            Path::new("/d/a-b-002-of-002.gguf")
        );
        assert_eq!(p("/d/one.gguf", 0, 1), Path::new("/d/one.gguf"));
        assert!(shard_path(Path::new("/d/a.gguf"), 1, 2).is_err());
        assert!(shard_path(Path::new("/d/a-x-of-2.gguf"), 1, 2).is_err());
    }

    #[test]
    fn read_kvs_skips_tokenizer_arrays_any_key_order() {
        fn key(b: &mut Vec<u8>, k: &str) {
            b.extend((k.len() as u64).to_le_bytes());
            b.extend(k.as_bytes());
        }
        fn i32_arr(b: &mut Vec<u8>, k: &str, n: u64) {
            key(b, k);
            b.extend(TYPE_ARRAY.to_le_bytes());
            b.extend(TYPE_I32.to_le_bytes());
            b.extend(n.to_le_bytes());
            for i in 0..n {
                b.extend((i as i32).to_le_bytes());
            }
        }
        let mut b = GGUF_MAGIC.to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend(4u64.to_le_bytes());
        i32_arr(&mut b, "a.attention.head_count_kv", 2);
        i32_arr(&mut b, "tokenizer.ggml.token_type", 150_000);
        key(&mut b, "tokenizer.ggml.tokens");
        b.extend(TYPE_ARRAY.to_le_bytes());
        b.extend(TYPE_STR.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        key(&mut b, "<s>");
        key(&mut b, "</s>");
        key(&mut b, "general.architecture");
        b.extend(TYPE_STR.to_le_bytes());
        key(&mut b, "a");
        let kvs = read_kvs(&mut std::io::Cursor::new(b)).unwrap();
        assert!(!kvs.iter().any(|(k, _)| k.starts_with("tokenizer.")));
        assert_eq!(
            kvs.iter()
                .find(|(k, _)| k == "a.attention.head_count_kv")
                .map(|(_, v)| v),
            Some(&KvVal::IntArr(vec![0, 1]))
        );
        assert_eq!(kv_str(&kvs, "general.architecture"), Some("a"));
    }
}
