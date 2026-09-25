//! T1 op-parity leg (M1.4): `ggml_mul_mat_id` over synthetic expert tensors,
//! recorded as a checksum-per-expert-id JSONL dump. Raw ggml FFI lives here
//! (design §13 unsafe policy): the CLI drives this through the safe
//! [`run_op_leg`] only.
//!
//! Shapes follow the MUL_MAT_ID contract at the vendor pin (`ggml.c`):
//! `as -> [k, m, n_expert]`, `b -> [k, 1, n_ids]`, `ids -> [1, n_ids] (i32)`,
//! `out -> [m, 1, n_ids]` — i.e. each of the `n_ids` fixed expert-id rows
//! routes through one expert and yields one output row of length `m`. The
//! per-id checksum is an f64 sum over that row. Weights and activations come
//! from a fixed-seed xorshift PRNG, so same-substrate legs are byte-identical
//! and cross-substrate legs differ only by floating-point implementation.
#![allow(unsafe_code)]

use std::path::Path;

use golem_llama_sys::{raw, substrate_backend};
use serde::Serialize;

use crate::atomic::write_atomic;
use crate::error::EngineError;

/// Rows per expert weight matrix (dump header `m`).
pub const OP_M: i64 = 256;
/// Dot-product depth (dump header `k`).
pub const OP_K: i64 = 128;
/// Expert matrices stacked in the weight tensor (dump header `n_expert`).
pub const OP_N_EXPERT: i64 = 8;
/// Routed expert-id rows (dump header `n_ids`).
pub const OP_N_IDS: u32 = 64;
/// Xorshift seed for both weights and activations (dump header `seed`).
pub const OP_SEED: u64 = 1234;

/// Fixed thread count for CPU-device compute (`ggml_backend_cpu_set_n_threads`).
/// Determinism of the dump comes from the fixed PRNG stream plus a fixed
/// per-row accumulation order, so the thread count must be part of the leg's
/// frozen inputs.
const N_THREADS: i32 = 2;

/// Context pool for the leg: tensor + graph *metadata* only — tensor data
/// lives in the device buffer (`no_alloc = true`). 1 MiB is orders of
/// magnitude above the metadata this leg needs (ggml aborts rather than
/// returning garbage when a pool is exhausted, so headroom is cheap).
const MEM_SIZE: u64 = 1024 * 1024;

/// Weight encoding of the expert tensors for one op-parity leg. The fp32 leg
/// computes the synthetic weights as-is; the q8_0 leg quantizes the *same*
/// weights with ggml's Q8_0 scheme (compute still accumulates in f32, exactly
/// as ggml does for Q8_0 x f32 MUL_MAT_ID).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    /// Expert weights stay `GGML_TYPE_F32`.
    Fp32,
    /// Expert weights quantized to `GGML_TYPE_Q8_0` via `ggml_quantize_chunk`.
    Q8_0,
}

impl Quant {
    /// Dump-header spelling of this variant.
    pub fn as_str(self) -> &'static str {
        match self {
            Quant::Fp32 => "fp32",
            Quant::Q8_0 => "q8_0",
        }
    }
}

/// Deterministic xorshift64 (Marsaglia 13/7/17) feeding both the weight and
/// activation fill; the seed is the frozen dump-contract `seed` field.
struct Xorshift(u64);

impl Xorshift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform sample in `[-1, 1)` from the top 24 bits (exact power-of-two
    /// scaling, so the mapping itself never loses bits).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// The synthetic leg inputs, from one frozen xorshift stream: expert weights
/// `[k, m, n_expert]` (row-major per expert: index `(expert * m + row) * k +
/// col`) first, then activations `[k, 1, n_ids]` (index `t * k + col`). Both
/// quant variants train these identical fp32 weights.
fn synthetic_inputs() -> (Vec<f32>, Vec<f32>) {
    let mut rng = Xorshift(OP_SEED);
    let weights: Vec<f32> = (0..(OP_M * OP_K * OP_N_EXPERT) as usize)
        .map(|_| rng.next_f32())
        .collect();
    let acts: Vec<f32> = (0..(OP_K * i64::from(OP_N_IDS)) as usize)
        .map(|_| rng.next_f32())
        .collect();
    (weights, acts)
}

/// Owned `ggml_context`; `Drop` calls `ggml_free` exactly once, mirroring the
/// session.rs RAII handle pattern (context outlives every tensor created in
/// it — dropping this last is what keeps the FFI pointers above valid).
struct GgmlContext(*mut raw::ggml_context);

impl Drop for GgmlContext {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: self.0 came from ggml_init and is freed exactly once here.
            unsafe { raw::ggml_free(self.0) };
        }
    }
}

/// Runs the T1 op-parity leg: computes the synthetic MUL_MAT_ID checksums
/// and writes the op-dump JSONL to `out` atomically (temp file + rename;
/// readers see old or new, never partial). Reads no model file and no user
/// input — the leg is fully synthetic.
pub fn run_op_leg(out: &Path, quant: Quant) -> Result<(), EngineError> {
    let (checksums, device) = compute_checksums(quant)?;
    let dump = render_op_dump(substrate_backend().as_str(), &device, quant, &checksums);
    write_atomic(out, &dump)?;
    Ok(())
}

/// Owned `ggml_backend`; `Drop` calls `ggml_backend_free` exactly once.
struct GgmlBackend(raw::ggml_backend_t);

impl Drop for GgmlBackend {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: self.0 came from ggml_backend_dev_init, freed once here.
            unsafe { raw::ggml_backend_free(self.0) };
        }
    }
}

/// Owned backend buffer from `ggml_backend_alloc_ctx_tensors_from_buft`; the
/// context does not own it, so `Drop` frees it exactly once (after the
/// `GgmlContext` field ordering below, tensors never outlive this).
struct GgmlBackendBuffer(raw::ggml_backend_buffer_t);

impl Drop for GgmlBackendBuffer {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: self.0 came from ggml_backend_alloc_ctx_tensors_from_buft.
            unsafe { raw::ggml_backend_buffer_free(self.0) };
        }
    }
}

/// Selects the executing device from the ggml backend registry: the first
/// GPU device when the binary is linked against a GPU backend (HIP), else
/// the CPU device. Returns the device handle and its lowercased registry
/// name (the dump header `device` field).
fn select_device() -> (raw::ggml_backend_dev_t, String) {
    for ty in [
        raw::ggml_backend_dev_type::GGML_BACKEND_DEVICE_TYPE_GPU,
        raw::ggml_backend_dev_type::GGML_BACKEND_DEVICE_TYPE_CPU,
    ] {
        // SAFETY: pure registry lookup; the returned handle is owned by the
        // registry and valid for the process lifetime.
        let dev = unsafe { raw::ggml_backend_dev_by_type(ty) };
        if dev.is_null() {
            continue;
        }
        // SAFETY: dev is a live registry device; the name is a static string.
        let name = unsafe { std::ffi::CStr::from_ptr(raw::ggml_backend_dev_name(dev)) }
            .to_string_lossy()
            .to_lowercase();
        return (dev, name);
    }
    unreachable!("ggml registry always registers a CPU device");
}

/// Computes the per-id checksums via raw ggml FFI through the backend
/// registry: the op graph is scheduled on the selected device (GPU when the
/// binary is linked against one, CPU otherwise) and returns the checksums
/// plus the executing device name. Test hook: the device probe alone is
/// re-exported as [`executing_device`].
fn compute_checksums(quant: Quant) -> Result<(Vec<f64>, String), EngineError> {
    let (dev, device) = select_device();

    // SAFETY: dev is a live registry device; null params select defaults.
    let backend = unsafe { raw::ggml_backend_dev_init(dev, std::ptr::null()) };
    if backend.is_null() {
        return Err(EngineError::OpCompute(format!(
            "ggml_backend_dev_init failed for device {device}"
        )));
    }
    let _backend = GgmlBackend(backend);
    // SAFETY: dev is a live registry device.
    if unsafe { raw::ggml_backend_dev_type(dev) }
        == raw::ggml_backend_dev_type::GGML_BACKEND_DEVICE_TYPE_CPU
    {
        // SAFETY: backend is a live CPU backend; the frozen thread count
        // pins the per-row accumulation order.
        unsafe { raw::ggml_backend_cpu_set_n_threads(backend, N_THREADS) };
    }

    let weight_type = match quant {
        Quant::Fp32 => raw::ggml_type::GGML_TYPE_F32,
        Quant::Q8_0 => raw::ggml_type::GGML_TYPE_Q8_0,
    };

    let params = raw::ggml_init_params {
        mem_size: MEM_SIZE,
        mem_buffer: std::ptr::null_mut(),
        no_alloc: true,
    };
    // SAFETY: params describe a self-owned metadata pool (null buffer ⇒ ggml
    // allocates); freed exactly once by the GgmlContext guard.
    let ctx = unsafe { raw::ggml_init(params) };
    if ctx.is_null() {
        return Err(EngineError::OpCompute(
            "ggml_init returned null (context pool allocation failed)".into(),
        ));
    }
    let _ctx = GgmlContext(ctx);

    let (weights_fp32, acts_fp32) = synthetic_inputs();

    // SAFETY: ctx is live with no_alloc=true, so each tensor's metadata is
    // pool-allocated while data stays unallocated until the device-buffer
    // allocation below; all tensors are dropped with the context.
    let weights = unsafe { raw::ggml_new_tensor_3d(ctx, weight_type, OP_K, OP_M, OP_N_EXPERT) };
    let acts = unsafe {
        raw::ggml_new_tensor_3d(
            ctx,
            raw::ggml_type::GGML_TYPE_F32,
            OP_K,
            1,
            i64::from(OP_N_IDS),
        )
    };
    let ids = unsafe {
        raw::ggml_new_tensor_2d(ctx, raw::ggml_type::GGML_TYPE_I32, 1, i64::from(OP_N_IDS))
    };
    for (what, ptr) in [("weights", weights), ("acts", acts), ("ids", ids)] {
        if ptr.is_null() {
            return Err(EngineError::OpCompute(format!(
                "ggml tensor allocation failed: {what}"
            )));
        }
    }

    // SAFETY: all three tensors are live in ctx and satisfy the shape
    // contract ggml_mul_mat_id asserts at this pin (module docs).
    let out = unsafe { raw::ggml_mul_mat_id(ctx, weights, acts, ids) };
    if out.is_null() {
        return Err(EngineError::OpCompute(
            "ggml_mul_mat_id returned null".into(),
        ));
    }

    // Allocate every ctx tensor (inputs + out) in one device buffer.
    // SAFETY: dev is a live registry device; ctx is live with all tensors
    // unallocated; the returned buffer is freed exactly once by the guard.
    let buft = unsafe { raw::ggml_backend_dev_buffer_type(dev) };
    let buffer = unsafe { raw::ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft) };
    if buffer.is_null() {
        return Err(EngineError::OpCompute(format!(
            "device-buffer allocation failed for device {device}"
        )));
    }
    let _buf = GgmlBackendBuffer(buffer);

    // Host-side payloads, uploaded verbatim: fp32 weights as raw bytes,
    // q8_0 weights quantized on the host into the tensor's exact size.
    // SAFETY: any &[f32] is valid to read as len * 4 u8s.
    let f32_bytes = |v: &[f32]| unsafe {
        std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
    };
    // SAFETY: tensor metadata is live; ggml_nbytes reads shape/type only.
    let weights_size = unsafe { raw::ggml_nbytes(weights) };
    let weights_bytes: Vec<u8> = match quant {
        Quant::Fp32 => f32_bytes(&weights_fp32).to_vec(),
        Quant::Q8_0 => {
            let mut q = vec![0u8; weights_size as usize];
            let nrows = OP_M * OP_N_EXPERT;
            // SAFETY: src is a live slice of nrows * n_per_row floats, dst is
            // exactly the quantized payload size, imatrix is null (Q8_0
            // requires none).
            let written = unsafe {
                raw::ggml_quantize_chunk(
                    raw::ggml_type::GGML_TYPE_Q8_0,
                    weights_fp32.as_ptr(),
                    q.as_mut_ptr().cast(),
                    0,
                    nrows,
                    OP_K,
                    std::ptr::null(),
                )
            };
            if written != weights_size {
                return Err(EngineError::OpCompute(format!(
                    "ggml_quantize_chunk wrote {written} bytes, tensor needs {weights_size}"
                )));
            }
            q
        }
    };
    let ids_vec: Vec<i32> = (0..i64::from(OP_N_IDS))
        .map(|t| (t % OP_N_EXPERT) as i32)
        .collect();

    for (tensor, data) in [
        (weights, weights_bytes.as_slice()),
        (acts, f32_bytes(&acts_fp32)),
        (ids, {
            // SAFETY: same &[i32]-as-bytes read as the f32 helper above.
            unsafe {
                std::slice::from_raw_parts(
                    ids_vec.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(&ids_vec),
                )
            }
        }),
    ] {
        // SAFETY: tensor is live and device-allocated; data is exactly
        // ggml_nbytes(tensor) bytes and the full tensor is written at
        // offset 0 before the graph runs.
        unsafe {
            raw::ggml_backend_tensor_set(tensor, data.as_ptr().cast(), 0, raw::ggml_nbytes(tensor));
        }
    }

    // SAFETY: ctx is live; the graph (metadata + node array) is allocated
    // from the context metadata pool.
    let graph = unsafe { raw::ggml_new_graph(ctx) };
    // SAFETY: graph is live and out is a ctx-owned tensor.
    unsafe { raw::ggml_build_forward_expand(graph, out) };
    // SAFETY: backend and graph are live; every graph input is
    // device-resident.
    let status = unsafe { raw::ggml_backend_graph_compute(backend, graph) };
    if status != raw::ggml_status::GGML_STATUS_SUCCESS {
        return Err(EngineError::OpCompute(format!(
            "ggml_backend_graph_compute reported {status:?} on device {device}"
        )));
    }

    // Download the output row set and checksum per routed id on the host.
    let out_len = (OP_M * i64::from(OP_N_IDS)) as usize;
    let mut out_host = vec![0.0f32; out_len];
    // SAFETY: out is live and computed; out_host is exactly
    // ggml_nbytes(out) bytes.
    unsafe {
        raw::ggml_backend_tensor_get(out, out_host.as_mut_ptr().cast(), 0, raw::ggml_nbytes(out));
    }
    let m = OP_M as usize;
    let mut checksums = Vec::with_capacity(OP_N_IDS as usize);
    for t in 0..OP_N_IDS as usize {
        let row = &out_host[t * m..(t + 1) * m];
        checksums.push(row.iter().map(|&v| f64::from(v)).sum::<f64>());
    }
    Ok((checksums, device))
}

/// The registry device the op graph is scheduled on (dump header `device`).
#[cfg(test)]
fn executing_device() -> String {
    select_device().1
}

/// Line 1 of an op dump. Field order is the serialization order and part of
/// the frozen dump contract (deliberately not key-sorted; mirrors the
/// `golem-parity-dump` header precedent).
#[derive(Serialize)]
struct OpDumpHeader<'a> {
    format: &'static str,
    version: u32,
    backend: &'a str,
    /// Backend that executed the op graph (registry device name from
    /// [`select_device`]).
    device: &'a str,
    m: i64,
    k: i64,
    n_expert: i64,
    n_ids: u32,
    seed: u64,
    quant: &'static str,
}

/// One per-expert-id record; key-sorted (checksum < id).
#[derive(Serialize)]
struct OpDumpRow {
    checksum: f64,
    id: u32,
}

/// Renders the dump body: line-1 header, then one record per expert id in
/// ascending id order, each a compact single-line JSON object in the frozen
/// field order.
fn render_op_dump(backend: &str, device: &str, quant: Quant, checksums: &[f64]) -> String {
    let header = OpDumpHeader {
        format: "golem-parity-op",
        version: 1,
        backend,
        device,
        m: OP_M,
        k: OP_K,
        n_expert: OP_N_EXPERT,
        n_ids: OP_N_IDS,
        seed: OP_SEED,
        quant: quant.as_str(),
    };
    let mut out = String::with_capacity(160 + 48 * checksums.len());
    // Plain string/number fields — serialization cannot fail.
    out.push_str(&serde_json::to_string(&header).expect("header serialization cannot fail"));
    out.push('\n');
    for (id, checksum) in checksums.iter().enumerate() {
        let record = OpDumpRow {
            checksum: *checksum,
            id: id as u32,
        };
        out.push_str(&serde_json::to_string(&record).expect("row serialization cannot fail"));
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "golem-engine-op-{}-{}-{tag}",
            std::process::id(),
            std::thread::current().name().unwrap_or("t")
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Runs the leg and parses the dump into `(dir, header, rows)`.
    fn run_and_parse(
        tag: &str,
        quant: Quant,
    ) -> (PathBuf, serde_json::Value, Vec<serde_json::Value>) {
        let dir = scratch_dir(tag);
        let dump = dir.join("op.jsonl");
        run_op_leg(&dump, quant).unwrap();
        let text = std::fs::read_to_string(&dump).unwrap();
        let mut lines = text.lines();
        let header: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        let rows = lines
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
            .collect::<Vec<_>>();
        (dir, header, rows)
    }

    /// The frozen header line, byte-exact.
    fn expected_header(quant: Quant) -> String {
        format!(
            "{{\"format\":\"golem-parity-op\",\"version\":1,\"backend\":\"{}\",\"device\":\"{}\",\"m\":256,\
             \"k\":128,\"n_expert\":8,\"n_ids\":64,\"seed\":1234,\"quant\":\"{}\"}}",
            substrate_backend().as_str(),
            executing_device(),
            quant.as_str(),
        )
    }

    /// Scalar reference: `out[t*m + r] = Σ_c weights[(id*m + r)*k + c] *
    /// acts[t*k + c]`. Returns per id the f64 row checksum and the row's L1
    /// magnitude `Σ_r |dot_r|` — the natural scale for quantization-noise
    /// bounds when the checksum itself nearly cancels.
    fn reference_row_stats(weights: &[f32], acts: &[f32]) -> Vec<(f64, f64)> {
        let m = OP_M as usize;
        let k = OP_K as usize;
        (0..OP_N_IDS as usize)
            .map(|t| {
                let id = t % OP_N_EXPERT as usize;
                let (mut sum, mut l1) = (0.0f64, 0.0f64);
                for r in 0..m {
                    let w_row = (id * m + r) * k;
                    let a_row = t * k;
                    let dot: f32 = (0..k).map(|c| weights[w_row + c] * acts[a_row + c]).sum();
                    sum += f64::from(dot);
                    l1 += f64::from(dot).abs();
                }
                (sum, l1)
            })
            .collect()
    }

    #[test]
    fn executing_device_matches_backend_class() {
        let device = executing_device();
        match substrate_backend() {
            golem_llama_sys::Backend::Cpu => {
                assert_eq!(device, "cpu", "CPU-linked substrate has no GPU device");
            }
            golem_llama_sys::Backend::Hip => {
                assert_ne!(
                    device, "cpu",
                    "HIP-linked substrate must schedule the op on the GPU device"
                );
            }
        }
    }

    #[test]
    fn quant_spelling_is_frozen() {
        assert_eq!(Quant::Fp32.as_str(), "fp32");
        assert_eq!(Quant::Q8_0.as_str(), "q8_0");
    }

    #[test]
    fn xorshift_is_deterministic_and_bounded() {
        let mut a = Xorshift(OP_SEED);
        let mut b = Xorshift(OP_SEED);
        for _ in 0..256 {
            assert_eq!(a.next_u64(), b.next_u64(), "same seed replays the stream");
        }
        let mut r = Xorshift(OP_SEED);
        for _ in 0..1024 {
            let v = r.next_f32();
            assert!((-1.0..1.0).contains(&v), "sample {v} outside [-1, 1)");
        }
    }

    #[test]
    fn xorshift_seed_changes_stream() {
        let mut a = Xorshift(OP_SEED);
        let mut b = Xorshift(OP_SEED + 1);
        let diff = (0..16).filter(|_| a.next_u64() != b.next_u64()).count();
        assert!(diff > 0, "distinct seeds must not share a stream prefix");
    }

    #[test]
    fn synthetic_inputs_shape_and_stream() {
        let (weights, acts) = synthetic_inputs();
        assert_eq!(weights.len(), (OP_M * OP_K * OP_N_EXPERT) as usize);
        assert_eq!(acts.len(), (OP_K * i64::from(OP_N_IDS)) as usize);
        // Every value is finite and in [-1, 1): the fill never leaks a
        // non-initialized slot into the tensors.
        assert!(
            weights
                .iter()
                .all(|v| v.is_finite() && (-1.0..1.0).contains(v))
        );
        assert!(
            acts.iter()
                .all(|v| v.is_finite() && (-1.0..1.0).contains(v))
        );
        // Frozen stream: re-deriving replays bit-identical values.
        let (weights2, acts2) = synthetic_inputs();
        assert_eq!(weights, weights2);
        assert_eq!(acts, acts2);
    }

    #[test]
    fn fp32_checksums_match_pure_rust_reference() {
        // End-to-end layout proof: the ggml leg must agree with a scalar
        // fp32 reference over the same synthetic tensors (loose relative
        // bound — ggml uses vectorized f32 accumulation, the reference is
        // scalar, so bitwise equality is not expected).
        let (weights, acts) = synthetic_inputs();
        let (got, _) = compute_checksums(Quant::Fp32).unwrap();
        let want = reference_row_stats(&weights, &acts);
        assert_eq!(got.len(), OP_N_IDS as usize);
        for (t, (g, (w, _))) in got.iter().zip(&want).enumerate() {
            let scale = w.abs().max(1.0);
            assert!(
                (g - w).abs() <= 1e-3 * scale,
                "id {t}: ggml {g} vs reference {w}"
            );
        }
    }

    #[test]
    fn fp32_checksums_are_finite() {
        let (_, _, rows) = run_and_parse("finite", Quant::Fp32);
        assert_eq!(rows.len(), OP_N_IDS as usize);
        for row in &rows {
            let c = row["checksum"].as_f64().unwrap();
            assert!(c.is_finite(), "non-finite checksum in {row}");
        }
    }

    #[test]
    fn fp32_dump_is_byte_identical_across_runs() {
        let dir = scratch_dir("det-fp32");
        let a = dir.join("a.jsonl");
        let b = dir.join("b.jsonl");
        run_op_leg(&a, Quant::Fp32).unwrap();
        run_op_leg(&b, Quant::Fp32).unwrap();
        assert_eq!(
            std::fs::read(&a).unwrap(),
            std::fs::read(&b).unwrap(),
            "same-substrate fp32 legs must be byte-identical"
        );
    }

    #[test]
    fn q8_0_dump_is_byte_identical_across_runs() {
        let dir = scratch_dir("det-q8");
        let a = dir.join("a.jsonl");
        let b = dir.join("b.jsonl");
        run_op_leg(&a, Quant::Q8_0).unwrap();
        run_op_leg(&b, Quant::Q8_0).unwrap();
        assert_eq!(
            std::fs::read(&a).unwrap(),
            std::fs::read(&b).unwrap(),
            "same-substrate q8_0 legs must be byte-identical"
        );
    }

    #[test]
    fn q8_0_differs_from_fp32_but_tracks_it() {
        let (_, h32, rows32) = run_and_parse("cmp-fp32", Quant::Fp32);
        let (_, h8, rows8) = run_and_parse("cmp-q8", Quant::Q8_0);
        assert_eq!(h32["quant"], "fp32");
        assert_eq!(h8["quant"], "q8_0");
        let differs = (0..OP_N_IDS as usize).any(|i| {
            rows32[i]["checksum"].as_f64().unwrap() != rows8[i]["checksum"].as_f64().unwrap()
        });
        assert!(differs, "quantized weights must change the checksums");
        // Q8_0 keeps ~7 mantissa bits per weight (≤ 1/254 relative rounding),
        // so each row dot product drifts by a small fraction of its own
        // magnitude. The honest bound is the row's L1 magnitude: the f64
        // checksum may nearly cancel (|checksum| << Σ|dot|), making a
        // relative-to-checksum bound meaningless. Bound: 2% of L1 (~4x the
        // worst-case per-weight rounding over a 128-term dot).
        let (weights, acts) = synthetic_inputs();
        let l1 = reference_row_stats(&weights, &acts);
        for i in 0..OP_N_IDS as usize {
            let a = rows32[i]["checksum"].as_f64().unwrap();
            let b = rows8[i]["checksum"].as_f64().unwrap();
            let scale = l1[i].1.max(1.0);
            assert!(
                (a - b).abs() <= 2e-2 * scale,
                "id {i}: q8_0 {b} drifted from fp32 {a} (row L1 {scale})"
            );
        }
    }

    #[test]
    fn dump_contract_is_frozen() {
        for quant in [Quant::Fp32, Quant::Q8_0] {
            let tag = format!("contract-{}", quant.as_str());
            let (dir, header, rows) = run_and_parse(&tag, quant);
            let text = std::fs::read_to_string(dir.join("op.jsonl")).unwrap();
            let first = text.lines().next().unwrap();
            assert_eq!(first, expected_header(quant), "byte-exact header line");

            assert_eq!(header["format"], "golem-parity-op");
            assert_eq!(header["version"], 1);
            assert_eq!(header["backend"], substrate_backend().as_str());
            assert_eq!(header["m"], 256);
            assert_eq!(header["k"], 128);
            assert_eq!(header["n_expert"], 8);
            assert_eq!(header["n_ids"], 64);
            assert_eq!(header["seed"], 1234);

            assert_eq!(rows.len(), OP_N_IDS as usize);
            for (i, row) in rows.iter().enumerate() {
                assert_eq!(row["id"], i as u64, "ids ascend from 0");
                assert!(row["checksum"].is_number());
                let line = format!("{row}");
                let (c, id) = (
                    line.find("\"checksum\"").unwrap(),
                    line.find("\"id\"").unwrap(),
                );
                assert!(c < id, "row records are key-sorted: checksum < id");
            }
        }
    }

    #[test]
    fn run_leaves_no_tmp_residue() {
        let dir = scratch_dir("atomic");
        let target = dir.join("dump.jsonl");
        run_op_leg(&target, Quant::Fp32).unwrap();
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            1,
            "no tmp residue next to the dump"
        );
    }
}
