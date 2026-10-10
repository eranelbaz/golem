//! R-8 HW leg: start-time model check on the real 6-shard model, sampled vs
//! `--full-verify`, each from a cold page cache. Writes the numbers that go
//! into `docs/baselines/m4.7-start-verify.md`. Never loads the model.
use std::time::Instant;

#[test]
#[ignore = "real model; GOLEM_HW=1 GOLEM_HW_MODEL=<shard 1> --run-ignored"]
fn hw_start_verify_sampled_vs_full_cold() {
    assert_eq!(
        std::env::var("GOLEM_HW").as_deref(),
        Ok("1"),
        "set GOLEM_HW=1"
    );
    let model: std::path::PathBuf = std::env::var("GOLEM_HW_MODEL")
        .expect("GOLEM_HW_MODEL must name shard 1")
        .into();
    let manifest = model.with_file_name("manifest.json");
    let m = golem_store::load_manifest(&manifest).expect("manifest");
    let evict = || {
        for s in &m.shards {
            let p = model.with_file_name(&s.file);
            let f = std::fs::File::open(&p).unwrap();
            assert!(
                golem_store::sys::drop_page_cache(&f, 0, s.bytes as usize),
                "{p:?}"
            );
        }
    };
    // Cold-run preconditions (Evidence): no golem instance maps a shard,
    // and `fincore` reports 0 resident pages for each after the eviction.
    let cold = || {
        for s in &m.shards {
            let p = model.with_file_name(&s.file);
            let users = std::process::Command::new("fuser")
                .arg(&p)
                .output()
                .expect("fuser");
            let pids = String::from_utf8_lossy(&users.stdout);
            assert!(pids.trim().is_empty(), "{p:?} mapped by pid(s) {pids}");
            let out = std::process::Command::new("fincore")
                .args(["--raw", "--noheadings", "--output", "RES"])
                .arg(&p)
                .output()
                .expect("fincore");
            let res = String::from_utf8_lossy(&out.stdout);
            assert_eq!(res.trim(), "0", "{p:?} resident bytes after eviction");
            println!("HW precondition {p:?}: fincore RES=0, no mapping pid");
        }
    };
    for full in [false, true] {
        evict();
        cold();
        let t = Instant::now();
        let warnings = golem_engine::verify::verify_model(&model, &manifest, full).expect("verify");
        println!(
            "HW start_verify full={full} shards={} cold_s={:.2} warnings={}",
            m.shards.len(),
            t.elapsed().as_secs_f64(),
            warnings.len()
        );
    }
}
