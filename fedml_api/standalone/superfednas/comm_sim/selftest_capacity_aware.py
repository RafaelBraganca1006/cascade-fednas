"""Local self-test for the capacity_aware sampler (Phase 3). No torch, no data,
no GPU: torch/wandb are stubbed out and the server model is faked with synthetic
parameter counts, mirroring the comm_sim self-test approach.

Run from the repo root:
    python3 fedml_api/standalone/superfednas/comm_sim/selftest_capacity_aware.py

Checks:
  1. Equivalence: with ample bandwidth for everyone, capacity_aware produces
     exactly the same assignments as TS_all_random (bandwidth is a ceiling,
     not a target).
  2. Feasibility: every assigned payload fits the client's bandwidth budget,
     except the documented best-effort cases (max-slot fallback / min tier).
  3. Maxnet slot: always goes to a client that can afford the max subnet, and
     rotates among eligible clients over rounds (fairness heap still works).
  4. Infeasible round: when nobody can afford the max subnet, the slot falls
     back to the fastest client (vanilla failure mode) and the counter ticks.
"""

import sys
import os
import types

# ---- stub torch and wandb before importing the server model ----
_torch = types.ModuleType("torch")
sys.modules.setdefault("torch", _torch)
_wandb = types.ModuleType("wandb")
_wandb.run = None
_wandb_sdk = types.ModuleType("wandb.sdk")
_wandb_lib = types.ModuleType("wandb.sdk.lib")


class _RunDisabled(object):
    pass


_wandb_lib.RunDisabled = _RunDisabled
_wandb_sdk.lib = _wandb_lib
_wandb.sdk = _wandb_sdk
sys.modules.setdefault("wandb", _wandb)
sys.modules.setdefault("wandb.sdk", _wandb_sdk)
sys.modules.setdefault("wandb.sdk.lib", _wandb_lib)

import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_bsm_path = os.path.join(
    _here, "..", "Server", "ServerModel", "base_server_model.py"
)
spec = importlib.util.spec_from_file_location("base_server_model", _bsm_path)
bsm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsm)

import numpy as np

MBIT = 1e6


class FakeParam(object):
    def __init__(self, n):
        self._n = n
        self.requires_grad = True

    def numel(self):
        return self._n


class FakeNet(object):
    def __init__(self, num_params):
        self._params = [FakeParam(num_params)]

    def parameters(self):
        return list(self._params)


class FakeClientModel(object):
    def __init__(self, config, num_params):
        self.model_config = config
        self.model = FakeNet(num_params)


# 4 tiers; size d in {1,2,3,4}; params = d * 250k -> payload = d * 8 Mbit
PARAMS_PER_D = 250_000
TIERS = {"0": {"d": 1}, "1": {"d": 2}, "2": {"d": 3}, "3": {"d": 4}}
MAX_D = 4
MIN_D = 1


class FakeServerModel(bsm.BaseServerModel):
    def init_model(self, init_params):
        return FakeNet(PARAMS_PER_D * MAX_D)

    def get_subnet(self, **kwargs):
        return FakeClientModel(dict(kwargs), PARAMS_PER_D * kwargs["d"])

    def is_max_net(self, arch):
        return arch["d"] == MAX_D

    def is_min_net(self, arch):
        return arch["d"] == MIN_D

    def max_subnet_arch(self):
        return {"d": MAX_D}

    def min_subnet_arch(self):
        return {"d": MIN_D}

    def random_subnet_arch(self):
        return {"d": int(np.random.randint(MIN_D, MAX_D + 1))}

    def random_depth_subnet_arch(self):
        return self.random_subnet_arch()

    def random_compound_subnet_arch(self):
        return self.random_subnet_arch()

    def add_subnet(self, shared_param_sum, shared_param_count, w_local):
        pass

    def active_subnet_index(self):
        pass

    def mutate_sample(self, sample_arch, mut_prob):
        pass


NUM_CLI = 8
SAMPLER_ARGS = {"diverse_subnets": TIERS, "client_per_round": NUM_CLI, "K": 1}
DEADLINE = 20.0


def payload_bits(d):
    return PARAMS_PER_D * d * 4 * 8


def make_server(method):
    s = FakeServerModel(None, method, NUM_CLI)
    s.load_client_sample_counts({i: 100 + i for i in range(NUM_CLI)})
    s.set_top_bottom_k(1, 1)
    return s


def run_round(server, round_num, bw=None):
    server.set_cli_indices(list(range(NUM_CLI)))
    if bw is not None:
        server.set_capacity_info(bw, DEADLINE, SAMPLER_ARGS)
    server.update_sample()
    out = []
    for idx in range(NUM_CLI):
        m = server.sample_subnet(round_num, idx, idx, SAMPLER_ARGS)
        out.append(m.model_config)
    return out


def test_equivalence_with_ts():
    ts = make_server("TS_all_random")
    cap = make_server("capacity_aware")
    ample = {i: 1e9 for i in range(NUM_CLI)}  # 1 Gbps: everything fits
    for r in range(30):
        a = run_round(ts, r)
        b = run_round(cap, r, bw=ample)
        assert a == b, "round %d: TS %s != capacity_aware %s" % (r, a, b)
    print("[ok] equivalence: ample bandwidth -> identical to TS_all_random")


def test_payloads_fit():
    cap = make_server("capacity_aware")
    # mixed bandwidths: 0.4 Mbps can't even fit the min tier (8 Mbit / 20 s)
    bw = {0: 50e6, 1: 10e6, 2: 2e6, 3: 0.4e6, 4: 50e6, 5: 10e6, 6: 2e6, 7: 50e6}
    clamped_seen = 0
    for r in range(50):
        configs = run_round(cap, r, bw=bw)
        clamped_seen += cap.cap_num_clamped
        for cli in range(NUM_CLI):
            d = configs[cli]["d"]
            budget = bw[cli] * DEADLINE
            if cli in cap.largest_subnet_min_idx:
                assert payload_bits(d) <= budget, (
                    "max slot to ineligible client %d" % cli
                )
            elif payload_bits(d) > budget:
                # only allowed when even the smallest tier doesn't fit
                assert d == MIN_D and payload_bits(MIN_D) > budget, (
                    "round %d client %d: d=%d doesn't fit and isn't the "
                    "min-tier best-effort case" % (r, cli, d)
                )
    assert clamped_seen > 0, "mixed bandwidth should clamp at least sometimes"
    print(
        "[ok] feasibility: all payloads fit (or are the documented best-effort "
        "min-tier case); clamped %d times" % clamped_seen
    )


def test_max_slot_eligible_and_rotates():
    cap = make_server("capacity_aware")
    # clients 0,1,4,7 can afford the max subnet (need >= 1.6 Mbps... use clear margins)
    bw = {0: 50e6, 1: 50e6, 2: 0.5e6, 3: 0.5e6, 4: 50e6, 5: 0.5e6, 6: 0.5e6, 7: 50e6}
    eligible = {0, 1, 4, 7}
    max_holders = set()
    for r in range(40):
        run_round(cap, r, bw=bw)
        holders = set(cap.largest_subnet_min_idx)
        assert holders and holders.issubset(eligible), (
            "round %d: max slot %s outside eligible %s" % (r, holders, eligible)
        )
        max_holders |= holders
    assert max_holders == eligible, (
        "max slot did not rotate across all eligible clients: %s" % max_holders
    )
    print("[ok] max slot: always eligible, rotated over all of %s" % eligible)


def test_infeasible_fallback():
    cap = make_server("capacity_aware")
    bw = {i: 0.1e6 for i in range(NUM_CLI)}
    bw[3] = 0.2e6  # fastest, still can't afford the max subnet
    run_round(cap, 0, bw=bw)
    assert cap.cap_max_infeasible == 1, "infeasible counter should tick"
    assert cap.largest_subnet_min_idx == {3}, (
        "fallback should pick the fastest client, got %s"
        % cap.largest_subnet_min_idx
    )
    print("[ok] infeasible round: counter ticked, fallback to fastest client")


def test_off_switch():
    cap = make_server("capacity_aware")
    ts = make_server("TS_all_random")
    for r in range(10):
        a = run_round(ts, r)
        b = run_round(cap, r)  # no set_capacity_info at all
        assert a == b
    print("[ok] comm_sim off: capacity_aware falls back to TS behaviour")


if __name__ == "__main__":
    test_equivalence_with_ts()
    test_payloads_fit()
    test_max_slot_eligible_and_rotates()
    test_infeasible_fallback()
    test_off_switch()
    print("all capacity_aware self-tests passed")
