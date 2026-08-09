#!/usr/bin/env python3
"""sweep_deadline -- drop and waste against the round deadline, for every
method family, computed offline.

WHY THIS IS OFFLINE AND WHY THAT IS NOT A SHORTCUT
==================================================
Drop and waste are decided by (payload, bandwidth, deadline) BEFORE a single
gradient is computed. They do not depend on the trained weights at all. So the
whole T-sweep is arithmetic over the seeded bandwidth chains, and running 4
methods x 6 deadlines on the cluster (24 runs, ~5 days serial) would spend a
week re-measuring a function we can evaluate exactly.

Precedent for the fidelity: analysis/pick_deadline.py predicted 50.8% dead slots
at T=20 and the real run reported drop_rate=0.508. This script goes further and
**validates itself** -- it recomputes T=20 for every method and compares against
the measured runs, printing the residual. If the residual is not ~0 the sweep is
not trustworthy and the script says so.

**It calls the PRODUCTION transmission code** (comm_sim.transmission's
all_or_nothing / truncate / truncate_stream) over plans built by the PRODUCTION
plan builders, driven by the PRODUCTION sampler with the trainer's own client
selection. Nothing here is a re-implementation of the mechanism; if it were, the
figure would be measuring this script rather than the system.

WHAT IS REPLAYED EXACTLY
    - client selection: np.random.seed(round); np.random.choice(20, 8, False)
      (superfednas_trainer._client_sampling)
    - architecture sampling: the real ServerResnet_10_26 with its
      cli_subnet_track state and update_sample() forced min/max slots, so the
      per-round arch distribution matches a real run round for round
    - bandwidth: MarkovBandwidth.tiered_continuous, same seed, same jitter

WHAT IS ASSUMED (and it is exact for these arms)
    The architecture draw does not depend on T. Sampling is seeded by
    (round + client_idx) and the deadline enters only at transmit time, so one
    replay serves every deadline in the sweep. This is true for the blind
    TS_all_random arms measured here; it is NOT true for capacity_aware /
    capacity_rank, whose assignment reads the bandwidth budget -- those samplers
    are deliberately excluded.

Run on the cluster (needs torch for the supernet); writes a small CSV that the
figure script reads locally.

    python analysis/sweep_deadline.py --out analysis/csv/deadline_sweep.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
# ServerModel/__init__ imports ServerDarts, whose model.py uses the ABSOLUTE
# import `from darts.operations import *`. train.py puts the elastic_nn
# directory on sys.path for exactly this reason (its line 34); without it the
# import of any ServerModel fails, even though nothing here touches DARTS.
sys.path.append(os.path.join(
    _ROOT, "fedml_api", "standalone", "superfednas", "elastic_nn"))

# --- ofa / torch compat shim -------------------------------------------------
# Identical to the one at the top of
# fedml_experiments/standalone/superfednas/train.py, and required for the same
# reason: the pinned `ofa` build still does `from torch._six import ...`, which
# torch >= 1.8 removed. Any module that reaches ServerModel triggers it, so the
# shim has to be installed BEFORE the first ofa import.
#
# Duplicated rather than imported because importing train.py would pull in its
# argparse, wandb and dataset machinery. If a third caller ever needs it, move
# it into a small module both import instead of copying it a third time.
import torch  # noqa: E402

if not hasattr(torch, "_six"):
    import types as _types
    import queue as _queue
    import collections.abc as _abc
    _six = _types.ModuleType("torch._six")
    _six.queue = _queue
    _six.string_classes = (str,)
    _six.int_classes = (int,)
    _six.container_abcs = _abc
    sys.modules["torch._six"] = _six
# --- end compat shim ---------------------------------------------------------

from fedml_api.standalone.superfednas.comm_sim.bandwidth import (  # noqa: E402
    MarkovBandwidth,
    assign_connectivity_tiers,
)
from fedml_api.standalone.superfednas.comm_sim.transmission import (  # noqa: E402
    all_or_nothing,
    build_clipped_plan,
    clip_arch_config,
    truncate,
    truncate_stream,
)
from fedml_api.standalone.superfednas.comm_sim.sharding import (  # noqa: E402
    build_block_stream,
)

# The canonical config, identical to every arm in results/README.md.
NUM_CLIENTS = 20
PER_ROUND = 8
ROUNDS = 150
BW_SEED = 0
JITTER = 0.45
BYTES_PER_PARAM = 4
TIERS = {
    "0": {"d": [0, 0, 0, 0], "e": 0.1},
    "1": {"d": [0, 1, 0, 1], "e": 0.14},
    "2": {"d": [1, 1, 1, 1], "e": 0.18},
    "3": {"d": [2, 2, 2, 2], "e": 0.25},
}

# Measured at T=20 by the real runs, used to validate the offline model.
# (drop %, waste %) -- results/README.md.
MEASURED_T20 = {
    "FedNAS-DARTS": (95.0, 85.0, "e6b_104209"),
    "SuperFedNAS (all-or-nothing)": (77.2, 63.6, "e6b_104127"),
    "FedAvg": (48.9, 23.1, "fedavg_104464"),
    "DynFedNAS (block stream)": (0.0, 13.5, "e6b_104125"),
}


def selected_clients(round_idx):
    """superfednas_trainer._client_sampling, replayed bit for bit."""
    np.random.seed(round_idx)
    return list(np.random.choice(range(NUM_CLIENTS), PER_ROUND, replace=False))


def build_bandwidth():
    """The trainer's bandwidth model. At corr=0 the connectivity-tier split is a
    pure function of (seed, n, fractions) -- assign_connectivity_tiers scales the
    label centre-of-mass by corr -- so dummy class counts give the identical
    split a real corr=0 run gets (the argument pick_deadline.py already makes)."""
    dummy = {c: np.ones(10, dtype=np.int64) for c in range(NUM_CLIENTS)}
    tiers = assign_connectivity_tiers(dummy, corr=0.0, seed=BW_SEED)
    return MarkovBandwidth.tiered_continuous(
        num_clients=NUM_CLIENTS, num_rounds=ROUNDS, client_tiers=tiers,
        seed=BW_SEED, jitter_sigma=JITTER,
    )


def build_server_model():
    """The real ServerResnet_10_26, so the arch draw and the forced min/max
    sandwich slots match a real run exactly."""
    from fedml_api.standalone.superfednas.Server.ServerModel import (
        ServerResnet_10_26,
    )

    params = {"n_classes": 10}
    server = ServerResnet_10_26(params, "TS_all_random", NUM_CLIENTS, False)
    # update_sample() computes max/second-max sample-count indices for EVERY
    # sampler, even though only `max_sample_count` reads them, and dereferences
    # this table unconditionally. TS_all_random never uses the result, so the
    # values are irrelevant -- but they must exist. Uniform is also the truthful
    # choice here: CIFAR-10 over 20 clients at alpha=100 is an even split.
    server.load_client_sample_counts([50000 // NUM_CLIENTS] * NUM_CLIENTS)
    server.set_top_bottom_k(1, 1)  # --top_k_maxnet / --bottom_k_maxnet defaults
    return server


def payload_bits(client_model):
    """CommSimulator.payload_bits' rule, kept identical."""
    n = sum(p.numel() for p in client_model.model.parameters() if p.requires_grad)
    return n * BYTES_PER_PARAM * 8


def replay_plans(rounds=ROUNDS):
    """One pass over the run: for every (round, client) produce the objects the
    three transmission modes need. Deadline-independent, so one replay serves
    the whole sweep."""
    server = build_server_model()
    sampler_args = {"diverse_subnets": TIERS, "client_per_round": PER_ROUND}
    tier_cfgs = [TIERS[k] for k in sorted(TIERS)]

    # get_subnet() materialises a model and dominates the runtime (~0.5 s each,
    # 9 per client). Clipped configs repeat heavily across clients -- every
    # client's tier-0 clip is the same minimum config -- so memoising by the
    # config makes the replay tractable. The cache is keyed on the config, and
    # get_subnet is a pure function of it, so this changes nothing but speed.
    payload_cache = {}
    stream_cache = {}

    def cfg_key(cfg):
        return repr(sorted((k, tuple(v) if isinstance(v, list) else v)
                           for k, v in cfg.items()))

    def cached_payload(cfg):
        k = cfg_key(cfg)
        if k not in payload_cache:
            payload_cache[k] = payload_bits(server.get_subnet(**cfg))
        return payload_cache[k]

    out = []
    for rnd in range(rounds):
        clients = selected_clients(rnd)
        server.set_cli_indices(clients)
        server.update_sample()  # forced min/max slots, stateful across rounds
        for idx, cli in enumerate(clients):
            cm = server.sample_subnet(rnd, cli, idx, sampler_args)
            arch = cm.model_config
            full = payload_bits(cm)
            plan = build_clipped_plan(arch, tier_cfgs, cached_payload)
            clipped = [clip_arch_config(arch, t) for t in tier_cfgs]
            skey = tuple(cfg_key(c) for c in clipped)
            if skey not in stream_cache:
                models = [server.get_subnet(**c) for c in clipped]
                stream_cache[skey] = build_block_stream(
                    models, clipped, bytes_per_param=BYTES_PER_PARAM,
                )
            out.append((rnd, cli, full, plan, stream_cache[skey]))
        if rnd % 25 == 0:
            print("  replay round %d/%d (cache: %d cfgs, %d streams)"
                  % (rnd, rounds, len(payload_cache), len(stream_cache)),
                  flush=True)
    return out


def sweep(plans, bw, deadlines, fixed_payloads):
    """For each method and deadline, aggregate drop and waste exactly as
    CommSimulator does: drop = slots delivering nothing / slots, waste =
    wasted bits / attempted bits."""
    rows = []
    for T in deadlines:
        acc = {}

        def add(method, report):
            a = acc.setdefault(method, dict(drop=0, n=0, att=0.0, wst=0.0))
            a["n"] += 1
            a["att"] += report.bits_attempted
            a["wst"] += report.bits_wasted
            if report.delivers_nothing():
                a["drop"] += 1

        for rnd, cli, full, plan, stream in plans:
            b = bw.bandwidth(cli, rnd)
            budget = b * T
            # fixed-payload single-model methods
            for name, pay in fixed_payloads.items():
                add(name, all_or_nothing(pay, budget, b, T))
            # the supernet arms
            add("SuperFedNAS (all-or-nothing)",
                all_or_nothing(full, budget, b, T))
            add("DynFedNAS (nested)", truncate(plan, budget, b, T))
            add("DynFedNAS (block stream)",
                truncate_stream(stream, budget, b, T))

        for method, a in acc.items():
            rows.append({
                "method": method,
                "deadline_s": T,
                "drop_pct": 100.0 * a["drop"] / max(a["n"], 1),
                "waste_pct": 100.0 * a["wst"] / max(a["att"], 1e-9),
                "attempted_gb": a["att"] / 8e9,
            })
        print("  T=%-6g done" % T, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadlines", type=float, nargs="+",
                    default=[10, 15, 20, 30, 40, 60, 80, 100, 120])
    ap.add_argument("--rounds", type=int, default=ROUNDS,
                    help="replay fewer rounds (smoke only -- the CSV is "
                         "only valid at the full 150)")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "csv", "deadline_sweep.csv"))
    args = ap.parse_args()

    print("replaying the run (arch draws are deadline-independent)...")
    plans = replay_plans(args.rounds)
    bw = build_bandwidth()

    # Fixed-payload baselines. Sized from the real models, not from the logs.

    from fedml_api.model.cv.resnet_gn import resnet18
    from fedml_api.standalone.superfednas.elastic_nn.darts.model import NetworkCIFAR
    from fedml_api.standalone.superfednas.elastic_nn.darts import genotypes

    fedavg = resnet18(num_classes=10)
    fednas = NetworkCIFAR(128, 10, 20, False, genotypes.FedNAS_V1)
    fixed = {
        "FedAvg": sum(p.numel() for p in fedavg.parameters()
                      if p.requires_grad) * BYTES_PER_PARAM * 8,
        "FedNAS-DARTS": sum(p.numel() for p in fednas.parameters()
                            if p.requires_grad) * BYTES_PER_PARAM * 8,
    }
    for k, v in fixed.items():
        print("  %-14s payload %.1f MB" % (k, v / 8e6))

    deadlines = sorted(set(list(args.deadlines) + [20.0]))
    print("sweeping %d deadlines..." % len(deadlines))
    rows = sweep(plans, bw, deadlines, fixed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "method", "deadline_s", "drop_pct", "waste_pct", "attempted_gb"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["method"], r["deadline_s"])):
            w.writerow(r)
    print("CSV -> %s" % args.out)

    # ---- self-validation against the measured runs -------------------------
    print("\nVALIDATION at T=20 (offline vs the real GPU runs)")
    print("%-30s %-18s %-18s" % ("method", "drop  off/meas", "waste off/meas"))
    worst = 0.0
    for r in rows:
        if r["deadline_s"] != 20.0 or r["method"] not in MEASURED_T20:
            continue
        md, mw, job = MEASURED_T20[r["method"]]
        dd, dw = r["drop_pct"] - md, r["waste_pct"] - mw
        worst = max(worst, abs(dd), abs(dw))
        print("%-30s %6.1f /%6.1f (%+.1f)  %6.1f /%6.1f (%+.1f)  [%s]" % (
            r["method"], r["drop_pct"], md, dd, r["waste_pct"], mw, dw, job))
    print()
    if worst <= 1.0:
        print("PASS -- worst residual %.2f pp. The offline model reproduces the "
              "runs; the sweep is trustworthy." % worst)
    else:
        print("FAIL -- worst residual %.2f pp. Do NOT plot this: the offline "
              "model and the runs disagree, and the disagreement has to be "
              "explained before the curve means anything." % worst)


if __name__ == "__main__":
    main()
