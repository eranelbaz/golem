# M1.4 hardware session notes (2026-09-24)

- Binaries: target-cpu/release/golem (GOLEM_LLAMA_SYS_BACKEND=cpu, GGML_CPU_REPACK=OFF), target-hip/release/golem (auto→hip, GGML_HIP=ON, gfx1100), same commit as 3260229. Both cmake caches verified (GGML_CPU_REPACK:BOOL=OFF; GGML_HIP:BOOL=ON).
- HIP substrate build works on this box (CMake 4.4.3 + ROCm); M1.3's reported blocker no longer reproduces.
- Leg dumps + logs: .m1-work/m14/{calibration,parity}/ (gitignored).
- T1 q8_0 bound rationale: measured max_abs 3.24e-2 is GPU MMQ vs CPU kernel accumulation-order difference over checksums of magnitude ~1e2 (relative ~3e-4); 1e-1 frozen as tripwire.
- T2 divergence shape: 6/8 prompts, first divergence positions [37,25,31,1,22,-,-,4]; gaps 0.04–0.35. Not an engine bug candidate per se — same 25.0% as M0.4 llama-cli text compare.
