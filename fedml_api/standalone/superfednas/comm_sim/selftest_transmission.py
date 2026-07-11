"""Local self-test for the v2 transmission layer (Stage 0). No torch, no data,
no GPU — transmission.py is pure bookkeeping and CommSimulator only needs a
fake bandwidth model and fake client models.

Run from the repo root:
    python3 fedml_api/standalone/superfednas/comm_sim/selftest_transmission.py

Checks:
  1. Legacy equivalence: CommSimulator.transmit() in all_or_nothing mode makes
     the exact success/drop decisions and success upload times of the v1
     simulator (bits/bw <= deadline), for a grid of payloads and bandwidths.
  2. round_time fix: a round with any drop lasts the full deadline; a round
     where every upload completed lasts the slowest completion time.
  3. Truncation: complete shells count, a half-arrived shell is wire waste,
     the delivered tier is the largest complete prefix.
  4. Boundaries: budget exactly at a shell edge delivers the shell; zero
     budget delivers nothing; ample budget completes the plan with zero waste.
  5. Monotonicity: more budget never delivers a smaller tier.
  6. Anytime accounting: in nested mode, delivered bits are always usable
     (waste only ever comes from the cut shell's remainder).
"""

import os
import sys
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tx = _load("transmission", "transmission.py")
# comm_simulator does a relative import of transmission; load it as a package
sys.path.insert(0, os.path.join(_here, "..", "..", "..", ".."))
from fedml_api.standalone.superfednas.comm_sim import CommSimulator  # noqa: E402

MBIT = 1e6


class FixedBandwidth(object):
    def __init__(self, bw_by_client):
        self.bw_by_client = bw_by_client

    def bandwidth(self, client_idx, round_num):
        return self.bw_by_client[client_idx]


class FakeParam(object):
    def __init__(self, n):
        self._n = n
        self.requires_grad = True

    def numel(self):
        return self._n


class FakeModel(object):
    def __init__(self, num_params):
        self._params = [FakeParam(num_params)]

    def parameters(self):
        return self._params


class FakeClientModel(object):
    def __init__(self, num_params):
        self.model = FakeModel(num_params)


def make_plan(tier_mbits=(8, 16, 32, 64), train_tier=3, order=None):
    tiers = [(mb * MBIT, {"tier": i}) for i, mb in enumerate(tier_mbits)]
    return tx.TransmissionPlan(tiers, train_tier=train_tier, order=order)


def test_legacy_equivalence():
    deadline = 20.0
    cases = []  # (payload_bits, bw_bps)
    for payload_mb in (1, 10, 40, 100, 400):
        for bw_mbps in (2, 10, 50):
            cases.append((payload_mb * MBIT, bw_mbps * MBIT))
    sim = CommSimulator(
        FixedBandwidth({i: bw for i, (_, bw) in enumerate(cases)}),
        deadline_s=deadline,
        tx_mode="all_or_nothing",
    )
    sim.start_round(0)
    for i, (bits, bw) in enumerate(cases):
        success, t = sim.transmit(0, i, FakeClientModel(int(bits // 32)), False)
        # v1 semantics: upload_time = bits/bw, success iff <= deadline
        v1_success = (bits / bw) <= deadline
        assert success == v1_success, (bits, bw)
        if success:
            assert abs(t - bits / bw) < 1e-9
    sim.end_round()
    print("PASS legacy equivalence (%d cases)" % len(cases))


def test_round_time():
    deadline = 20.0
    bw = FixedBandwidth({0: 50 * MBIT, 1: 50 * MBIT, 2: 1 * MBIT})
    # payloads: 100 Mbit at 50 Mbps = 2 s; 400 Mbit at 50 = 8 s; 100 Mbit at 1 Mbps = drop
    small = FakeClientModel(int(100 * MBIT // 32))
    big = FakeClientModel(int(400 * MBIT // 32))

    sim = CommSimulator(bw, deadline_s=deadline)
    sim.start_round(0)  # all complete -> slowest completion
    sim.transmit(0, 0, small, False)
    sim.transmit(0, 1, big, False)
    s = sim.end_round()
    assert abs(s["comm/round_time_s"] - 8.0) < 1e-9, s["comm/round_time_s"]

    sim.start_round(1)  # one drop -> full deadline (the v2 fix)
    sim.transmit(1, 0, small, False)
    sim.transmit(1, 2, small, False)  # 100 Mbit at 1 Mbps: 100 s > 20 s
    s = sim.end_round()
    assert s["comm/round_time_s"] == deadline, s["comm/round_time_s"]
    assert s["comm/num_dropped"] == 1
    # wire waste of the dropper = bw x deadline = 20 Mbit
    assert abs(s["comm/MB_wasted_round"] - (20 * MBIT / 8e6)) < 1e-9
    print("PASS round_time fix")


def test_truncation_and_boundaries():
    plan = make_plan()  # shells: 8, 8, 16, 32 Mbit; tiers cumulative: 8/16/32/64
    window = 20.0

    def deliver(budget_mbit, bw_mbps=10):
        return tx.truncate(plan, budget_mbit * MBIT, bw_mbps * MBIT, window)

    r = deliver(0)  # zero budget: nothing
    assert r.delivers_nothing() and r.bits_delivered == 0

    r = deliver(4)  # cut inside shell 0: nothing usable, 4 Mbit wasted
    assert r.delivers_nothing() and abs(r.bits_wasted - 4 * MBIT) < 1e-9

    r = deliver(8)  # exactly shell 0: tier S delivered, zero waste
    assert r.delivered_tier == 0 and r.bits_wasted == 0

    r = deliver(20)  # shells 0+1 complete (16), 4 Mbit into shell 2 wasted
    assert r.delivered_tier == 1
    assert abs(r.bits_delivered - 16 * MBIT) < 1e-9
    assert abs(r.bits_wasted - 4 * MBIT) < 1e-9

    r = deliver(64)  # full plan: completed, zero waste
    assert r.delivered_tier == 3 and r.completed_plan and r.bits_wasted == 0

    # out-of-order shells: arriving shell 1 without shell 0 completes no tier
    plan_top_first = make_plan(order=[1, 0, 2, 3])
    r = tx.truncate(plan_top_first, 8 * MBIT, 10 * MBIT, window)
    assert r.delivers_nothing() and r.delivered_tier == -1
    r = tx.truncate(plan_top_first, 16 * MBIT, 10 * MBIT, window)
    assert r.delivered_tier == 1  # shells {1,0} arrived -> prefix 0..1 complete
    print("PASS truncation + boundaries + ordering")


def test_clip_arch_config():
    # middle-client arch: per-stage d, per-block e (as TS_all_random samples)
    arch = {"d": [0, 1, 0, 2], "e": [0.25, 0.1, 0.18, 0.14], "w": [None, None]}
    tier1 = {"d": [0, 1, 0, 1], "e": 0.14}
    clipped = tx.clip_arch_config(arch, tier1)
    assert clipped["d"] == [0, 1, 0, 1]
    assert clipped["e"] == [0.14, 0.1, 0.14, 0.14]
    assert clipped["w"] == [None, None]  # passthrough
    # clipping by the max tier returns the arch itself
    xl = {"d": [2, 2, 2, 2], "e": 0.25}
    same = tx.clip_arch_config(arch, xl)
    assert same["d"] == arch["d"] and same["e"] == arch["e"]
    # nestedness of the clipped ladder: elementwise monotone in the tier
    tiers = [
        {"d": [0, 0, 0, 0], "e": 0.1},
        {"d": [0, 1, 0, 1], "e": 0.14},
        {"d": [1, 1, 1, 1], "e": 0.18},
        {"d": [2, 2, 2, 2], "e": 0.25},
    ]
    prev = None
    for t in tiers:
        c = tx.clip_arch_config(arch, t)
        if prev is not None:
            assert all(a >= b for a, b in zip(c["d"], prev["d"]))
            assert all(a >= b for a, b in zip(c["e"], prev["e"]))
        prev = c
    print("PASS clip_arch_config")


def test_build_clipped_plan():
    tiers = [
        {"d": [0, 0, 0, 0], "e": 0.1},
        {"d": [0, 1, 0, 1], "e": 0.14},
        {"d": [1, 1, 1, 1], "e": 0.18},
        {"d": [2, 2, 2, 2], "e": 0.25},
    ]
    arch = {"d": [0, 1, 0, 2], "e": [0.25, 0.1, 0.18, 0.14], "w": [None]}

    # fake payload: monotone in the config (as real param counts are)
    def payload(cfg):
        e = cfg["e"] if isinstance(cfg["e"], (list, tuple)) else [cfg["e"]] * 4
        return (sum(cfg["d"]) + 10 * sum(e)) * MBIT

    plan = tx.build_clipped_plan(arch, tiers, payload)
    assert plan.train_tier == 3
    # top of the clipped ladder is α itself (completing the plan == delivering α)
    top_cfg = plan.tiers[-1][1]
    assert top_cfg["d"] == arch["d"] and top_cfg["e"] == arch["e"]
    bits = [b for b, _ in plan.tiers]
    assert bits == sorted(bits), "clipped ladder must be ascending"
    # every rung is the clip of α by that tier
    for (b, cfg), t in zip(plan.tiers, tiers):
        assert cfg == tx.clip_arch_config(arch, t)
    print("PASS build_clipped_plan")


def test_saturating_ladder():
    # a small trained arch: clipped tiers saturate -> zero-size top shells
    tiers = [
        (8 * MBIT, {"tier": 0}),
        (12 * MBIT, {"tier": 1}),
        (12 * MBIT, {"tier": 2}),  # == tier 1: alpha already inside tier 1
        (12 * MBIT, {"tier": 3}),
    ]
    plan = tx.TransmissionPlan(tiers, train_tier=3)
    r = tx.truncate(plan, 12 * MBIT, 10 * MBIT, 20.0)
    assert r.delivered_tier == 3 and r.completed_plan and r.bits_wasted == 0
    r = tx.truncate(plan, 8 * MBIT, 10 * MBIT, 20.0)
    assert r.delivered_tier == 0
    print("PASS saturating clipped ladder")


def test_monotonicity():
    plan = make_plan()
    window = 20.0
    prev_tier = -1
    for budget_mbit in range(0, 70, 1):
        r = tx.truncate(plan, budget_mbit * MBIT, 10 * MBIT, window)
        assert r.delivered_tier >= prev_tier, budget_mbit
        prev_tier = r.delivered_tier
    print("PASS monotonicity")


def test_nested_sim_accounting():
    deadline = 20.0
    # client 0: 10 Mbps x 20 s = 200 Mbit budget -> full 64 Mbit plan
    # client 1: 1 Mbps x 20 s = 20 Mbit budget  -> tiers 0+1 (16), 4 wasted
    # client 2: 0.2 Mbps x 20 s = 4 Mbit budget -> nothing, 4 wasted
    sim = CommSimulator(
        FixedBandwidth({0: 10 * MBIT, 1: 1 * MBIT, 2: 0.2 * MBIT}),
        deadline_s=deadline,
        tx_mode="nested",
    )
    sim.start_round(0)
    r0 = sim.transmit_plan(0, 0, make_plan(), is_max_net=True)
    r1 = sim.transmit_plan(0, 1, make_plan(), is_max_net=False)
    r2 = sim.transmit_plan(0, 2, make_plan(), is_max_net=False)
    s = sim.end_round()
    assert r0.completed_plan and r1.delivered_tier == 1 and r2.delivers_nothing()
    assert s["comm/num_dropped"] == 1  # only the deliver-nothing client
    assert s["comm/participation_rate"] == 2.0 / 3.0
    assert abs(s["comm/MB_sent_round"] - (64 + 16) * MBIT / 8e6) < 1e-9
    assert abs(s["comm/MB_wasted_round"] - (4 + 4) * MBIT / 8e6) < 1e-9
    assert s["comm/max_subnet_failed"] == 0  # max holder completed its plan
    assert s["comm/round_time_s"] == deadline  # truncations -> full deadline
    print("PASS nested simulator accounting")


if __name__ == "__main__":
    test_legacy_equivalence()
    test_round_time()
    test_truncation_and_boundaries()
    test_clip_arch_config()
    test_build_clipped_plan()
    test_saturating_ladder()
    test_monotonicity()
    test_nested_sim_accounting()
    print("ALL TRANSMISSION SELF-TESTS PASSED")
