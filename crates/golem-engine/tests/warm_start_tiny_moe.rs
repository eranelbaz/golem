//! Warm restart (§6.3 warm-up, §12.3): a pooled session saves its §6.2
//! policy words and step; the next session restores them verbatim, pre-fills
//! the RAM pool in the background before any request, and decodes exactly
//! what a cold session decodes (T3/I1). A corrupt file starts cold with a
//! warning. Pooled tiny-moe through the engine-registered hooks. The
//! restore leg needs HIP: the routing readback that books the words fires
//! only on a GPU expert split.
mod common;

use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use golem_engine::config::EngineConfig;
use golem_engine::session::Session;

fn tmp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("golem-warm-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn pooled(warm_state_dir: Option<PathBuf>) -> EngineConfig {
    EngineConfig {
        model_path: common::tiny_moe_path(),
        ram_experts_gib: 1,
        device_cache_experts: 4,
        n_predict: 16,
        warm_state_dir,
        ..EngineConfig::default()
    }
}

#[cfg(feature = "hip-tests")]
fn greedy(s: &mut Session) -> String {
    use golem_engine::session::ChatTurn;
    let mut out = String::new();
    s.chat_turn(
        &[ChatTurn {
            role: "user",
            content: "the quick brown fox",
        }],
        &mut |p| out.push_str(p),
    )
    .unwrap();
    assert!(!out.is_empty(), "greedy turn generated nothing");
    out
}

fn words(s: &Session) -> (u64, Vec<(usize, u64, u64)>) {
    s.p3_hooks().expect("pooled").warm_words_for_test()
}

#[cfg(feature = "hip-tests")]
#[test]
fn saved_state_restores_verbatim_prefills_and_decodes_like_cold() {
    let dir = tmp_dir("restore");
    let _ = std::fs::remove_dir_all(&dir);

    let cold_out;
    let cold_fill;
    let saved;
    {
        let mut a = Session::new(&pooled(Some(dir.clone()))).expect("session A");
        assert_eq!(
            words(&a),
            (0, Vec::new()),
            "self-check books nothing (§6.2)"
        );
        // The load self-check already pages its routed units in: the
        // pre-fill must beat this cold baseline, not just 0.
        cold_fill = a.p3_hooks().unwrap().pool_fill();
        cold_out = greedy(&mut a);
        saved = words(&a);
        assert!(saved.0 > 0 && !saved.1.is_empty(), "turn booked: {saved:?}");
        a.save_warm_state().expect("save");
    }
    let fp = golem_placement::scores::model_fingerprint(&common::tiny_moe_path()).unwrap();
    let text = std::fs::read_to_string(dir.join(format!("scores-{fp}.txt"))).unwrap();
    assert!(text.starts_with("golem-scores v1 "), "{text}");

    let warm_out = {
        let mut b = Session::new(&pooled(Some(dir.clone()))).expect("session B");
        assert_eq!(words(&b), saved, "step and words restored verbatim");
        let hooks = b.p3_hooks().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !hooks.warm_fill_done() {
            assert!(
                std::time::Instant::now() < deadline,
                "warm pre-fill never completed"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        // Completed before the request: every planned unit is resident.
        let planned = hooks.warm_planned_for_test();
        assert!(!planned.is_empty(), "restored state planned no pre-fill");
        let cold: Vec<_> = planned
            .iter()
            .filter(|&&(l, e)| !hooks.is_resident_for_test(l, e))
            .collect();
        assert!(
            cold.is_empty(),
            "pre-fill left planned units cold: {cold:?}"
        );
        assert!(
            hooks.pool_fill() > cold_fill,
            "pre-fill beat the cold {cold_fill}"
        );
        greedy(&mut b)
    };
    let fresh_out = {
        let mut c = Session::new(&pooled(None)).expect("cold session");
        greedy(&mut c)
    };
    assert_eq!(fresh_out, cold_out, "cold sessions agree");
    assert_eq!(warm_out, cold_out, "warm start changes no output (T3/I1)");
}

#[derive(Clone)]
struct Buf(Arc<Mutex<Vec<u8>>>);

impl Write for Buf {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn corrupt_or_mismatched_file_starts_cold_with_a_warning() {
    let logs = Buf(Arc::new(Mutex::new(Vec::new())));
    let w = logs.clone();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_ansi(false)
        .with_writer(move || w.clone())
        .init();
    let dir = tmp_dir("corrupt");
    let fp = golem_placement::scores::model_fingerprint(&common::tiny_moe_path()).unwrap();
    for (name, body) in [
        ("garbage", "not a score file\n".to_owned()),
        (
            "mismatch",
            "golem-scores v1 layers=1 experts=8 half_life=512 step=3 keys=0 model=0000000000000000\n"
                .to_owned(),
        ),
    ] {
        let case = dir.join(name);
        std::fs::create_dir_all(&case).unwrap();
        std::fs::write(case.join(format!("scores-{fp}.txt")), body).unwrap();
        logs.0.lock().unwrap().clear();
        let s = Session::new(&pooled(Some(case))).expect("a bad file never fails the load");
        assert_eq!(words(&s), (0, Vec::new()), "{name}: cold");
        assert_eq!(s.p3_hooks().unwrap().prefetch_stats().prefetch_pageins, 0);
        let log = String::from_utf8_lossy(&logs.0.lock().unwrap()).into_owned();
        assert!(
            log.contains("warm-start scores refused; starting cold"),
            "{name}: {log}"
        );
    }
}
