//! mlock slot arena acceptance (§5.4): locked pages, disjoint slots.

use golem_placement::pool::Arena;

#[test]
fn arena_locks_and_slots_are_disjoint() {
    let slot_size = 16 * 32 * 4; // n_ff_exp * n_embd * F32 (tiny-moe plane)
    let a = Arena::new(slot_size * 4, slot_size).expect("mlock may fail on low RLIMIT_MEMLOCK");
    for i in 0..a.slot_count() {
        a.fill_slot(i, &[i as u8; 16 * 32 * 4]);
    }
    for i in 0..a.slot_count() {
        assert!(a.slot(i).iter().all(|&b| b == i as u8));
    }
}

#[test]
fn arena_rejects_zero() {
    assert!(Arena::new(0, 4096).is_err());
}
