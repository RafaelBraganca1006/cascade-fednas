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
import pickle
import sys

import numpy as np

def _find_code_root():
    """Locate the directory that holds ``fedml_api``.

    This script exists in two places -- ``analysis/`` at the repository root
    (where the figures live) and ``dynfednas_v2/analysis/`` (what the cluster
    checks out) -- and the code root sits at a different depth in each. Probing
    for it lets ONE file serve both, which matters more than it sounds: the two
    copies had already drifted apart once, and a stale duplicate of a script
    whose whole job is to produce numbers fails silently, by printing older
    numbers rather than by crashing.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, ".."),
                 os.path.join(here, "..", "dynfednas_v2")):
        cand = os.path.abspath(cand)
        if os.path.isdir(os.path.join(cand, "fedml_api")):
            return cand
    raise RuntimeError("cannot find fedml_api from %s" % here)


_ROOT = _find_code_root()
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
#
# Skipped when torch is absent, which is a supported way to run this script:
# only replay_plans() needs it, and `--plans` reads a pickle produced by an
# earlier replay. comm_sim.transmission and comm_sim.sharding -- everything the
# sweep itself touches -- import nothing beyond numpy.
try:
    import torch  # noqa: E402
except ImportError:
    torch = None
else:
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


def build_bandwidth(bw_seed=None):
    """The trainer's bandwidth model. At corr=0 the connectivity-tier split is a
    pure function of (seed, n, fractions) -- assign_connectivity_tiers scales the
    label centre-of-mass by corr -- so dummy class counts give the identical
    split a real corr=0 run gets (the argument pick_deadline.py already makes)."""
    seed = BW_SEED if bw_seed is None else bw_seed
    dummy = {c: np.ones(10, dtype=np.int64) for c in range(NUM_CLIENTS)}
    tiers = assign_connectivity_tiers(dummy, corr=0.0, seed=seed)
    return MarkovBandwidth.tiered_continuous(
        num_clients=NUM_CLIENTS, num_rounds=ROUNDS, client_tiers=tiers,
        seed=seed, jitter_sigma=JITTER,
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


def sweep(plans, bw_for, deadlines, fixed_payloads):
    """For each method and deadline, aggregate drop and waste exactly as
    CommSimulator does: drop = slots delivering nothing / slots, waste =
    wasted bits / attempted bits.

    ``bw_for(T) -> BandwidthModel``. It is a factory rather than a single model
    because the trace-replay model integrates the measured series over the round
    window, so a model built for one deadline is not valid at another. The
    synthetic factory ignores T and returns the same chain every time.

    Also accumulates ``wall_clock_h``: the campaign's communication wall-clock,
    summing CommSimulator's own round-time rule (the deadline if anything was
    still in flight when it expired, otherwise the last successful upload's
    finish time). That column is what makes "just raise the deadline" a
    quantified trade instead of a free win -- see the thesis, Section 4.2.
    """
    rows = []
    per_round = {}
    for rnd, cli, full, plan, stream in plans:
        per_round.setdefault(rnd, []).append((cli, full, plan, stream))

    for T in deadlines:
        bw = bw_for(T)
        acc = {}

        def add(method, report, rnd_acc):
            a = acc.setdefault(method, dict(drop=0, n=0, att=0.0, wst=0.0,
                                            wall=0.0))
            a["n"] += 1
            a["att"] += report.bits_attempted
            a["wst"] += report.bits_wasted
            if report.delivers_nothing():
                a["drop"] += 1
            m = rnd_acc.setdefault(method, dict(any_incomplete=False, tmax=0.0))
            if report.completed_plan:
                m["tmax"] = max(m["tmax"], report.time_used)
            else:
                m["any_incomplete"] = True

        for rnd in sorted(per_round):
            rnd_acc = {}
            for cli, full, plan, stream in per_round[rnd]:
                b = bw.bandwidth(cli, rnd)
                budget = b * T
                # fixed-payload single-model methods
                for name, pay in fixed_payloads.items():
                    add(name, all_or_nothing(pay, budget, b, T), rnd_acc)
                # the supernet arms
                add("SuperFedNAS (all-or-nothing)",
                    all_or_nothing(full, budget, b, T), rnd_acc)
                add("DynFedNAS (nested)", truncate(plan, budget, b, T), rnd_acc)
                add("DynFedNAS (block stream)",
                    truncate_stream(stream, budget, b, T), rnd_acc)
            for method, m in rnd_acc.items():
                acc[method]["wall"] += T if m["any_incomplete"] else m["tmax"]

        for method, a in acc.items():
            rows.append({
                "method": method,
                "deadline_s": T,
                "drop_pct": 100.0 * a["drop"] / max(a["n"], 1),
                "waste_pct": 100.0 * a["wst"] / max(a["att"], 1e-9),
                "attempted_gb": a["att"] / 8e9,
                "wall_clock_h": a["wall"] / 3600.0,
            })
        print("  T=%-6g done" % T, flush=True)
    return rows


def build_fixed_payloads():
    """Fixed-payload baselines, sized from the real models, not from the logs."""
    from fedml_api.model.cv.resnet_gn import resnet18
    from fedml_api.standalone.superfednas.elastic_nn.darts.model import NetworkCIFAR
    from fedml_api.standalone.superfednas.elastic_nn.darts import genotypes

    fedavg = resnet18(num_classes=10)
    fednas = NetworkCIFAR(128, 10, 20, False, genotypes.FedNAS_V1)
    return {
        "FedAvg": sum(p.numel() for p in fedavg.parameters()
                      if p.requires_grad) * BYTES_PER_PARAM * 8,
        "FedNAS-DARTS": sum(p.numel() for p in fednas.parameters()
                            if p.requires_grad) * BYTES_PER_PARAM * 8,
    }


def build_sprout_bandwidth_factory(num_clients, num_rounds, seed, archive):
    """A ``bw_for(T)`` over the Sprout UPLINK traces.

    Identical in shape to the Irish factory, and different in one way that
    matters: these are measurements of the direction the deadline actually
    constrains, so no ul_factor is applied and none is offered. There is nothing
    to correct for.
    """
    from fedml_api.standalone.superfednas.comm_sim.bandwidth import (
        TraceReplayBandwidth,
    )
    from fedml_api.standalone.superfednas.comm_sim.traces import (
        load_sprout_client_traces,
    )

    client_traces, offsets, which = load_sprout_client_traces(
        num_clients, seed=seed, path=archive)
    mix = {n: which.count(n) for n in sorted(set(which))}
    print("  uplink trace population: %s" % mix)
    allv = np.concatenate([client_traces[c] for c in client_traces])
    print("  measured uplink: median %.2f Mbit/s, p10 %.2f, p90 %.2f"
          % (np.median(allv) / 1e6, np.percentile(allv, 10) / 1e6,
             np.percentile(allv, 90) / 1e6))

    def bw_for(T):
        return TraceReplayBandwidth(
            client_traces, num_rounds=num_rounds, window_s=T,
            sample_hz=1.0, start_offsets=offsets)

    return bw_for


def build_trace_bandwidth_factory(num_clients, num_rounds, seed, patterns,
                                  ul_factor, archive):
    """A ``bw_for(T)`` over the measured Irish 4G traces.

    A fresh model per deadline, because the replay integrates the 1 Hz series
    over the round window: the T=20 model averages 20 samples per round, the
    T=120 model averages 120. The client->trace assignment and the start offsets
    are drawn from the seed alone, so every deadline in a sweep replays the SAME
    twenty measured links over the same stretch of real time, and the curve is a
    function of T and nothing else.
    """
    from fedml_api.standalone.superfednas.comm_sim.bandwidth import (
        TraceReplayBandwidth,
    )
    from fedml_api.standalone.superfednas.comm_sim.traces import (
        load_client_traces,
    )

    client_traces, offsets, which = load_client_traces(
        num_clients, seed=seed, patterns=patterns, ul_factor=ul_factor,
        path=archive)
    mix = {p: which.count(p) for p in sorted(set(which))}
    print("  trace population: %s  (ul_factor=%.2f)" % (mix, ul_factor))
    allv = np.concatenate([client_traces[c] for c in client_traces])
    print("  measured throughput over the drawn traces: p10 %.2f / p50 %.2f / "
          "p90 %.2f Mbps" % tuple(np.percentile(allv, [10, 50, 90]) / 1e6))

    def bw_for(T):
        return TraceReplayBandwidth(client_traces, num_rounds=num_rounds,
                                    window_s=T, sample_hz=1.0,
                                    start_offsets=offsets)
    return bw_for


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadlines", type=float, nargs="+",
                    default=[10, 15, 20, 30, 40, 60, 80, 100, 120])
    ap.add_argument("--rounds", type=int, default=ROUNDS,
                    help="replay fewer rounds (smoke only -- the CSV is "
                         "only valid at the full 150)")
    ap.add_argument("--bandwidth", choices=["synthetic", "trace", "sprout"],
                    default="synthetic",
                    help="synthetic = the canonical tiered_continuous chains; "
                         "trace = replay of the measured Irish 4G dataset")
    ap.add_argument("--trace-patterns", nargs="+", default=None,
                    choices=["static", "pedestrian", "car", "bus", "train"],
                    help="restrict the trace population to these mobility "
                         "patterns (default: all five)")
    ap.add_argument("--trace-seed", type=int, default=BW_SEED)
    ap.add_argument("--bw-seed", type=int, default=None,
                    help="synthetic channel seed. The sweep is exact given one "
                         "channel realization, so a confidence band over it "
                         "requires running several seeds and is not something "
                         "one run can report.")
    ap.add_argument("--ul-factor", type=float, default=1.0,
                    help="scale every measured sample; 1.0 = as measured "
                         "(downlink, optimistic for an uplink deadline), "
                         "0.5 = the pessimistic sensitivity")
    ap.add_argument("--trace-archive", default=None)
    # The replay is the only step that needs torch. Dumping it once lets the
    # sweep -- and every re-sweep under a different bandwidth model -- run
    # anywhere, since comm_sim.transmission and comm_sim.sharding are torch-free.
    ap.add_argument("--dump-plans", default=None,
                    help="replay, write the plans to this pickle, and stop")
    ap.add_argument("--plans", default=None,
                    help="load plans from a pickle instead of replaying "
                         "(no torch needed)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.plans:
        with open(args.plans, "rb") as fh:
            blob = pickle.load(fh)
        plans, fixed, dumped_rounds = (blob["plans"], blob["fixed"],
                                       blob["rounds"])
        if dumped_rounds != args.rounds:
            sys.exit("the pickle holds %d rounds but --rounds is %d; re-dump"
                     % (dumped_rounds, args.rounds))
        print("loaded %d plan slots from %s" % (len(plans), args.plans))
    else:
        print("replaying the run (arch draws are deadline-independent)...")
        plans = replay_plans(args.rounds)
        fixed = build_fixed_payloads()

    for k, v in fixed.items():
        print("  %-14s payload %.1f MB" % (k, v / 8e6))

    if args.dump_plans:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_plans)),
                    exist_ok=True)
        with open(args.dump_plans, "wb") as fh:
            pickle.dump(dict(plans=plans, fixed=fixed, rounds=args.rounds), fh)
        print("PLANS -> %s (%.1f MB)"
              % (args.dump_plans, os.path.getsize(args.dump_plans) / 1e6))
        return

    if args.bandwidth == "sprout":
        bw_for = build_sprout_bandwidth_factory(
            NUM_CLIENTS, args.rounds, args.trace_seed, args.trace_archive)
    elif args.bandwidth == "trace":
        bw_for = build_trace_bandwidth_factory(
            NUM_CLIENTS, args.rounds, args.trace_seed, args.trace_patterns,
            args.ul_factor, args.trace_archive)
    else:
        chain = build_bandwidth(args.bw_seed)
        bw_for = lambda T: chain  # noqa: E731 -- T-independent by construction

    out = args.out or os.path.join(
        here, "csv", "deadline_sweep%s.csv"
        % {"synthetic": "", "trace": "_trace", "sprout": "_sprout"}[args.bandwidth])

    deadlines = sorted(set(list(args.deadlines) + [20.0]))
    print("sweeping %d deadlines..." % len(deadlines))
    rows = sweep(plans, bw_for, deadlines, fixed)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "method", "deadline_s", "drop_pct", "waste_pct", "attempted_gb",
            "wall_clock_h"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["method"], r["deadline_s"])):
            w.writerow(r)
    print("CSV -> %s" % out)

    # ---- self-validation against the measured runs -------------------------
    # Only the synthetic model has GPU runs to be validated against. The trace
    # sweep is a DIFFERENT bandwidth process, so a residual against those runs
    # would be meaningless -- it is the mechanism code that is shared and
    # already validated, not the bandwidth.
    if args.bandwidth != "synthetic":
        print("\n(no validation table: the measured runs used the synthetic "
              "chains, so there is nothing to compare a trace sweep against. "
              "The transmission code being exercised is the same code the "
              "synthetic sweep validates to 0.10 pp.)")
        return

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
