"""Self-tests for the E6b selection layer + cap sampler (DIRECTION §3.2b).

Numpy-only, no torch -- runs anywhere:

    python3 fedml_api/standalone/superfednas/comm_sim/selftest_selection.py

Two of these are load-bearing rather than routine:

  test_cap_max_equals_blind -- makes the sentence "blind sampling is a member of
      the policy class" TRUE. That sentence is going in the thesis and is the
      whole justification for the cap-based action space; if the clip at the max
      tier is not the identity, or consumes different randomness, the claim is
      false and the E6b comparison inherits the same confound that invalidated
      E5/E6.

  test_legacy_selection_path_unchanged -- guards the np.random.seed(round_idx)
      call order in _client_sampling. Every archived run's client sequence
      depends on it; a refactor that consumes randomness before the draw would
      silently change WHICH clients participate in every round while still
      producing plausible accuracies, invalidating every cross-arm comparison in
      results/README.md.
"""

import importlib.util
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tx = _load("transmission", os.path.join(_here, "transmission.py"))
sel = _load("selection", os.path.join(_here, "policies", "selection.py"))

# The canonical --diverse_subnets ladder (E6b config).
TIERS = {
    "0": {"d": [0, 0, 0, 0], "e": 0.1},
    "1": {"d": [0, 1, 0, 1], "e": 0.14},
    "2": {"d": [1, 1, 1, 1], "e": 0.18},
    "3": {"d": [2, 2, 2, 2], "e": 0.25},
}
TIER_BITS = {"0": 41.6e6 * 8, "1": 98.6e6 * 8, "2": 137.6e6 * 8, "3": 286.9e6 * 8}


def random_subnet_arch():
    """The production sampler's draw (server_resnet_32x32_10_26), reproduced
    exactly: per-block expand, per-stage depth, same np.random call order."""
    depth_list = [0, 1, 2]
    expand_ratio_list = [0.1, 0.14, 0.18, 0.22, 0.25]
    expand_setting = [np.random.choice(expand_ratio_list) for _ in range(12)]
    depth_setting = [np.random.choice(depth_list) for _ in range(4)]
    return {"d": depth_setting, "e": expand_setting}


def test_cap_max_equals_blind():
    """cap = max tier must be the IDENTITY on both the arch AND the RNG."""
    cap = TIERS["3"]
    for seed in range(50):
        np.random.seed(seed)
        blind = random_subnet_arch()
        blind_next = np.random.random()  # RNG state AFTER the draw

        np.random.seed(seed)
        capped = tx.clip_arch_config(random_subnet_arch(), cap)
        capped_next = np.random.random()

        assert list(blind["d"]) == list(capped["d"]), (seed, blind["d"], capped["d"])
        assert list(blind["e"]) == list(capped["e"]), (seed, blind["e"], capped["e"])
        assert blind_next == capped_next, (
            "cap sampling consumed different randomness at seed %d -- 'blind is "
            "a member of the policy class' would be FALSE" % seed
        )
    print("PASS cap=XL reproduces blind bit-identically (arch + RNG state)")


def test_cap_below_max_actually_clips():
    """A lower cap must genuinely restrict, or the action does nothing."""
    for tier_id in ("0", "1", "2"):
        cap = TIERS[tier_id]
        restricted = 0
        for seed in range(50):
            np.random.seed(seed)
            arch = random_subnet_arch()
            np.random.seed(seed)
            capped = tx.clip_arch_config(random_subnet_arch(), cap)
            assert all(c <= cap["e"] for c in capped["e"]), (tier_id, capped["e"])
            assert all(
                c <= d for c, d in zip(capped["d"], cap["d"])
            ), (tier_id, capped["d"], cap["d"])
            if list(arch["e"]) != list(capped["e"]) or list(arch["d"]) != list(
                capped["d"]
            ):
                restricted += 1
        assert restricted > 25, (
            "cap %s barely restricted anything (%d/50)" % (tier_id, restricted)
        )
    print("PASS lower caps genuinely restrict (nested inside the cap)")


def test_legacy_selection_path_unchanged():
    """The uniform draw must stay bit-identical to every archived run.

    Ground truth is not a value I chose -- it is grepped from
    results/v2_arms/v2_arm_b_nested_baseline.out, whose first two rounds log
    `client_indexes = [18  1 19  8 10 17  6 13]` and `[ 3 16  6 10  2 14  4 17]`.
    If _client_sampling ever consumes randomness before its np.random.seed /
    np.random.choice pair, these change and every cross-arm comparison in
    results/README.md silently stops being like-for-like.
    """
    archived = {
        0: [18, 1, 19, 8, 10, 17, 6, 13],
        1: [3, 16, 6, 10, 2, 14, 4, 17],
        2: [12, 4, 18, 0, 9, 5, 3, 10],
    }
    for r, expected in archived.items():
        np.random.seed(r)
        got = [int(x) for x in np.random.choice(range(20), 8, replace=False)]
        assert got == expected, (
            "round %d client draw CHANGED: %s != archived %s" % (r, got, expected)
        )
    print("PASS legacy uniform client draw matches the archived runs")


def _ctx(bw, delivered=None, last_seen=None, min_bits=TIER_BITS["0"], window=40.0):
    return sel.SelectionContext(
        bw_true_bps=bw,
        bw_est_bps=bw,
        min_tier_bits=min_bits,
        upload_window_s=window,
        delivered_counts=delivered or {},
        last_round_seen=last_seen or {},
    )


def test_oracle_bandwidth_probe():
    bw = {c: (c + 1) * 1e6 for c in range(10)}  # client 9 fastest
    p = sel.OracleBandwidthSelector()
    chosen = p.select(0, list(range(10)), 4, _ctx(bw))
    assert chosen == [9, 8, 7, 6], chosen
    # deterministic under ties
    flat = {c: 5e6 for c in range(10)}
    assert p.select(0, list(range(10)), 3, _ctx(flat)) == [0, 1, 2]
    print("PASS probe A: top-K by bandwidth, deterministic ties")


def test_coverage_probe_prefers_starved_but_feasible():
    # everyone fast enough except clients 0 and 1 (dead this round)
    bw = {c: 50e6 for c in range(10)}
    bw[0] = 0.5e6
    bw[1] = 0.5e6
    delivered = {c: {"0": 5} for c in range(10)}
    delivered[0] = {}       # starved, but dead this round
    delivered[3] = {}       # starved AND feasible -> must be picked first
    delivered[4] = {"0": 1}
    p = sel.CoverageSelector()
    chosen = p.select(7, list(range(10)), 3, _ctx(bw, delivered))
    assert chosen[0] == 3, chosen  # starved + feasible wins
    assert 0 not in chosen, "picked a client that provably cannot deliver"
    assert chosen[1] == 4, chosen  # next least-delivered feasible
    print("PASS probe B: starved-but-feasible first, dead slots avoided (%s)" % chosen)


def test_coverage_probe_never_returns_short():
    """If fewer than K clients are feasible it must still fill the round."""
    bw = {c: 0.5e6 for c in range(6)}  # everybody dead
    p = sel.CoverageSelector()
    chosen = p.select(0, list(range(6)), 4, _ctx(bw))
    assert len(chosen) == 4 and len(set(chosen)) == 4, chosen
    print("PASS probe B fills the round even when nobody is feasible")


def test_is_dead_threshold():
    ctx = _ctx({0: 1e6, 1: 20e6}, window=40.0)  # 40 Mb vs 800 Mb budget
    assert ctx.is_dead(0) and not ctx.is_dead(1)
    print("PASS dead-slot threshold matches the smallest tier's payload")


def test_coverage_ranks_on_participation_not_tier_matches():
    """Regression for the delivered-counts undercount (fixed 2026-08-10).

    The coverage objective means "how much has this client actually got
    through". It used to be read off the TIER-keyed table, which only records
    deliveries that completed a configured ladder tier -- and under blind
    TS_all_random a middle client trains an arbitrary elastic architecture that
    matches no tier, so roughly half of all deliveries were invisible. Job
    104125 is a 150-round run at 0% drop, ~60 deliveries per client, and its log
    reported coverage_max=29.

    Client 0 here has delivered in many rounds but completed a tier only once;
    client 1 is the opposite. A selector ranking on tier matches would call
    client 0 the starved one and pick it. Ranking on participation must pick
    client 1.
    """
    ctx = sel.SelectionContext(
        bw_true_bps={0: 50e6, 1: 50e6},
        upload_window_s=20.0,
        min_tier_bits=1.0,
        delivered_counts={0: {"3": 1}, 1: {"0": 40, "1": 40}},
        delivery_rounds={0: 55, 1: 3},
        last_round_seen={0: 10, 1: 10},
    )
    chosen = sel.CoverageSelector().select(0, [0, 1], 1, ctx)
    assert chosen == [1], (
        "coverage selector ranked on tier matches, not participation: %s" % chosen
    )

    # Without the participation table it falls back to the old behaviour, so an
    # old caller still runs -- and demonstrably picks the wrong client.
    old = sel.SelectionContext(
        bw_true_bps={0: 50e6, 1: 50e6},
        upload_window_s=20.0,
        min_tier_bits=1.0,
        delivered_counts={0: {"3": 1}, 1: {"0": 40, "1": 40}},
        last_round_seen={0: 10, 1: 10},
    )
    assert sel.CoverageSelector().select(0, [0, 1], 1, old) == [0]
    print("PASS coverage ranks on participation, not on tier matches")


if __name__ == "__main__":
    test_cap_max_equals_blind()
    test_cap_below_max_actually_clips()
    test_legacy_selection_path_unchanged()
    test_oracle_bandwidth_probe()
    test_coverage_probe_prefers_starved_but_feasible()
    test_coverage_probe_never_returns_short()
    test_is_dead_threshold()
    test_coverage_ranks_on_participation_not_tier_matches()
    print("ALL PASS")
