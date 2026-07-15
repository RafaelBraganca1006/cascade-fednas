"""Self-test for tiered connectivity profiles + bandwidth-data correlation
(E4 groundwork). Numpy-only — runs without torch:

    python3 fedml_api/standalone/superfednas/comm_sim/selftest_tiered_bandwidth.py
"""

import importlib.util
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "bandwidth", os.path.join(_here, "bandwidth.py")
)
bw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bw)


def test_default_unchanged_by_refactor():
    """default() must equal the per-client path with the same single matrix."""
    a = bw.MarkovBandwidth.default(8, 60, seed=0)
    mbps = 1e6
    b = bw.MarkovBandwidth(
        8, 60, [50 * mbps, 10 * mbps, 2 * mbps],
        bw.MarkovBandwidth.TIER_TRANSITIONS[1], seed=0,
        per_client_transition=[bw.MarkovBandwidth.TIER_TRANSITIONS[1]] * 8,
    )
    assert (a._traj == b._traj).all(), "refactor changed default trajectories"
    c = bw.MarkovBandwidth.default(8, 60, seed=0)
    assert (a._traj == c._traj).all(), "default not deterministic"
    print("PASS default unchanged + deterministic")


def test_tiers_are_persistently_different():
    n, r = 30, 400
    tiers = [0] * 10 + [1] * 10 + [2] * 10
    m = bw.MarkovBandwidth.tiered(n, r, tiers, seed=0)
    mean_bw = [
        np.mean([m.bandwidth(c, t) for t in range(r)]) for c in range(n)
    ]
    poor = np.mean(mean_bw[:10]) / 1e6
    avg = np.mean(mean_bw[10:20]) / 1e6
    good = np.mean(mean_bw[20:]) / 1e6
    assert poor < avg < good, (poor, avg, good)
    assert poor < 15 and good > 30, (poor, good)
    print(f"PASS tier means Mbps: poor={poor:.1f} avg={avg:.1f} good={good:.1f}")


def _fake_counts(n, num_classes=10, seed=0):
    """Clients 0..n-1 concentrate class (c mod 10): dominant class known."""
    rng = np.random.default_rng(seed)
    counts = {}
    for c in range(n):
        base = rng.integers(0, 5, num_classes)
        base[c % num_classes] += 100
        counts[c] = base
    return counts


def test_corr_extremes():
    n = 20
    counts = _fake_counts(n)
    # corr=1: clients with low class center-of-mass must be tier 0 (poor)
    t1 = bw.assign_connectivity_tiers(counts, corr=1.0, seed=0)
    com = np.array([
        (counts[c] * np.arange(10)).sum() / counts[c].sum() for c in range(n)
    ])
    poor_com = com[[c for c in range(n) if t1[c] == 0]]
    good_com = com[[c for c in range(n) if t1[c] == 2]]
    assert poor_com.max() < good_com.min(), "corr=1 not class-sorted"
    # corr=0: seeded-random, independent of a shuffled copy of the data
    t0a = bw.assign_connectivity_tiers(counts, corr=0.0, seed=0)
    t0b = bw.assign_connectivity_tiers(_fake_counts(n, seed=99), corr=0.0, seed=0)
    assert t0a == t0b, "corr=0 must ignore the data"
    # fractions respected (0.3/0.4/0.3 of 20 = 6/8/6)
    for t in (t0a, t1):
        assert [t.count(k) for k in (0, 1, 2)] == [6, 8, 6], t
    # determinism + seed sensitivity
    assert t1 == bw.assign_connectivity_tiers(counts, corr=1.0, seed=0)
    assert t0a != bw.assign_connectivity_tiers(counts, corr=0.0, seed=1)
    print("PASS corr extremes, fractions, determinism")


def test_corr_interpolates():
    n = 40
    counts = _fake_counts(n)
    com = np.array([
        (counts[c] * np.arange(10)).sum() / counts[c].sum() for c in range(n)
    ])

    def poor_com_mean(corr):
        t = bw.assign_connectivity_tiers(counts, corr=corr, seed=3)
        return com[[c for c in range(n) if t[c] == 0]].mean()

    lo, mid, hi = poor_com_mean(0.0), poor_com_mean(0.5), poor_com_mean(1.0)
    assert hi < mid < lo or (hi < lo and abs(mid - hi) < abs(mid - lo) + 2), (
        lo, mid, hi,
    )  # poor tier's class center falls as corr rises (monotone in expectation)
    print(f"PASS interpolation: poor-tier class center {lo:.2f} -> {mid:.2f} -> {hi:.2f}")


if __name__ == "__main__":
    test_default_unchanged_by_refactor()
    test_tiers_are_persistently_different()
    test_corr_extremes()
    test_corr_interpolates()
    print("ALL PASS")
