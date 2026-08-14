"""Self-tests for measured-trace replay (TraceReplayBandwidth + traces.py).

Torch-free and archive-optional: the properties of the replay model are checked
against hand-built series, so they hold whether or not analysis/data/irish4g.npz
is present. The tests that need the real archive skip themselves when it is
missing and say so, rather than passing vacuously.

    python3 -m fedml_api.standalone.superfednas.comm_sim.selftest_traces
"""

import numpy as np

from .bandwidth import TraceReplayBandwidth
from . import traces as tr


def _check(name, cond, detail=""):
    print("  %-58s %s%s" % (name, "PASS" if cond else "FAIL",
                            "" if cond else "   <- " + detail))
    return bool(cond)


def test_window_is_the_mean():
    """budget = bandwidth * T must equal the bits the measured link carried.

    This is the whole reason bandwidth() returns a slice mean: the simulator
    multiplies it by the window, so mean*T reconstructs the integral exactly.
    """
    series = np.arange(1, 101, dtype=float) * 1e6  # 1..100 Mbps, 1 Hz
    T = 10
    bw = TraceReplayBandwidth({0: series}, num_rounds=10, window_s=T)
    ok = True
    for r in range(10):
        expect_bits = series[r * T:(r + 1) * T].sum()  # 1 s per sample
        got_bits = bw.bandwidth(0, r) * T
        ok &= abs(got_bits - expect_bits) < 1e-6
    return _check("round budget equals the integral of the measured series", ok)


def test_wraps_and_offsets():
    series = np.array([1.0, 2.0, 3.0, 4.0]) * 1e6
    bw = TraceReplayBandwidth({0: series}, num_rounds=8, window_s=1)
    seq = [bw.bandwidth(0, r) for r in range(8)]
    wrapped = _check("a trace shorter than the run wraps around",
                     seq == list(series) * 2)

    off = TraceReplayBandwidth({0: series}, num_rounds=4, window_s=1,
                               start_offsets={0: 2})
    shifted = _check("start_offsets shift the entry point into the trace",
                     [off.bandwidth(0, r) for r in range(4)]
                     == [3e6, 4e6, 1e6, 2e6])
    return wrapped and shifted


def test_order_independent():
    """The simulator asks for the same (c, r) twice per round -- once to size
    the update, once to charge for it -- and a disagreement between those two
    answers would silently bill a client at a bandwidth it was never given."""
    rng = np.random.default_rng(0)
    series = rng.lognormal(15, 1, 500)
    bw = TraceReplayBandwidth({0: series, 1: series[::-1].copy()},
                              num_rounds=20, window_s=7,
                              start_offsets={0: 3, 1: 11})
    forward = [(c, r, bw.bandwidth(c, r)) for c in (0, 1) for r in range(20)]
    backward = [(c, r, bw.bandwidth(c, r))
                for c in (1, 0) for r in reversed(range(20))]
    lookup = {(c, r): v for c, r, v in backward}
    return _check("bandwidth(c, r) does not depend on call order",
                  all(abs(lookup[(c, r)] - v) < 1e-12 for c, r, v in forward))


def test_longer_window_is_smoother():
    """Averaging over a longer deadline must reduce the spread of round
    bandwidths. If it did not, the windowing is not doing what it claims."""
    rng = np.random.default_rng(1)
    series = rng.lognormal(15, 1.2, 4000)
    sds = []
    for T in (5, 20, 80):
        bw = TraceReplayBandwidth({0: series}, num_rounds=40, window_s=T)
        sds.append(np.std([bw.bandwidth(0, r) for r in range(40)]))
    return _check("a longer deadline averages away round-to-round variation",
                  sds[0] > sds[1] > sds[2], "sds=%s" % [round(s) for s in sds])


def test_empty_trace_rejected():
    try:
        TraceReplayBandwidth({0: np.array([])}, num_rounds=2, window_s=1)
    except ValueError:
        return _check("an empty trace raises instead of producing nan", True)
    return _check("an empty trace raises instead of producing nan", False)


def test_seed_determinism():
    """Two loads at the same seed must draw the same links; a different seed
    must draw a different population, or the seed is not doing anything."""
    try:
        a, ao, aw = tr.load_client_traces(20, seed=0)
        b, bo, bw_ = tr.load_client_traces(20, seed=0)
        c, co, cw = tr.load_client_traces(20, seed=1)
    except FileNotFoundError as exc:
        print("  %-58s SKIP  (%s)" % ("seeded trace assignment is reproducible",
                                      str(exc).split("--")[0].strip()))
        return True
    same = all(np.array_equal(a[i], b[i]) for i in a) and ao == bo
    differs = any(not np.array_equal(a[i], c[i]) for i in a) or ao != co
    return (_check("same seed draws the same links and offsets", same)
            and _check("a different seed draws a different population", differs))


def test_no_replacement_and_patterns():
    try:
        ct, _off, which = tr.load_client_traces(20, seed=0)
    except FileNotFoundError as exc:
        print("  %-58s SKIP  (%s)" % ("20 clients get 20 distinct measured links",
                                      str(exc).split("--")[0].strip()))
        return True
    ids = {id(v.tobytes()) for v in ct.values()}
    distinct = len({v.tobytes() for v in ct.values()})
    ok = _check("20 clients get 20 distinct measured links", distinct == 20,
                "only %d distinct" % distinct)
    del ids
    sub, _o, subw = tr.load_client_traces(20, seed=0, patterns=["train"])
    ok &= _check("patterns= restricts the population",
                 set(subw) == {"train"}, "got %s" % set(subw))
    # ul_factor must scale, not reshape.
    half, _o2, _w2 = tr.load_client_traces(20, seed=0, ul_factor=0.5)
    ok &= _check("ul_factor scales every sample and nothing else",
                 all(np.allclose(half[i], ct[i] * 0.5) for i in ct))
    return ok


def main():
    print("trace replay self-tests")
    results = [
        test_window_is_the_mean(),
        test_wraps_and_offsets(),
        test_order_independent(),
        test_longer_window_is_smoother(),
        test_empty_trace_rejected(),
        test_seed_determinism(),
        test_no_replacement_and_patterns(),
    ]
    print()
    print("%d/%d checks passed" % (sum(bool(r) for r in results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
