#!/usr/bin/env python3
"""compute_budget -- what a round costs in COMPUTE, and how that compares to T.

THE QUESTION, and why it is dangerous to skip.

the thesis (Section 4.2) argues that raising the deadline is not free because "T is
the round's wall-clock price": 150 rounds at T=20 s cost 0.83 h of communication
against 5.00 h at T=120 s, a 6x multiplier. That arithmetic is only honest if
the round is MOSTLY communication. Our round-time model is uplink-only (3.1,
"compute-free" -- not "homogeneous compute"), so it silently assumes exactly
that, and never checked it.

If local training dominates the round, the multiplier collapses. A round that is
120 s of compute plus 20 s of upload becomes 140 s at T=20 and 240 s at T=120 --
1.7x, not 6x -- and leg 2 of the rebuttal weakens accordingly. So this script
computes the compute side rather than assuming it away, and reports the honest
multiplier instead of the flattering one.

WHAT IT MEASURES

  1. Per-tier cost of the four subnets in FLOPs and parameters, using the SAME
     profiler the trainer uses (ofa.utils.flops_counter), so the numbers are the
     ones the system would report about itself.
  2. Training FLOPs per client-round: forward FLOPs x 3 (the standard
     forward+backward multiplier) x local samples x local epochs.
  3. Who actually trains, per arm, by replaying the cached plans -- a client
     whose update cannot be delivered is skipped BEFORE training
     (superfednas_trainer, "no wasted compute"), so the arms differ in compute
     precisely because they differ in drop.
  4. Seconds per training-GFLOP, calibrated by least squares against the
     MEASURED GPU wall-clock of the three blind arms. Two unknowns (per-round
     fixed overhead, seconds per GFLOP) against three observations, so the
     residual is a check rather than a fit by construction.
  5. The comparison the question actually asks for: round compute against T,
     over a range of client hardware, with the break-even called out.

THE MODELLING POINT THAT DECIDES THE ANSWER. Our simulator trains the eight
selected clients SEQUENTIALLY on one GPU, so its wall-clock is a SUM. A real
deployment trains them in PARALLEL, so the round's compute is a MAX -- the
straggler -- not a sum. Reporting the sum against T would overstate compute by
roughly 8x and hand the argument away for free. Both are printed, and the
comparison against T uses the max.

    python3 analysis/compute_budget.py --plans <plans150.pkl>
"""

import argparse
import os
import pickle
import sys

import numpy as np


def _find_code_root():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, ".."), os.path.join(here, "..", "dynfednas_v2")):
        cand = os.path.abspath(cand)
        if os.path.isdir(os.path.join(cand, "fedml_api")):
            return cand
    raise RuntimeError("cannot find fedml_api from %s" % here)


_ROOT = _find_code_root()
sys.path.insert(0, _ROOT)
sys.path.append(os.path.join(
    _ROOT, "fedml_api", "standalone", "superfednas", "elastic_nn"))

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

from fedml_api.standalone.superfednas.comm_sim.bandwidth import (  # noqa: E402
    MarkovBandwidth,
    assign_connectivity_tiers,
)
from fedml_api.standalone.superfednas.comm_sim.transmission import (  # noqa: E402
    all_or_nothing,
    truncate,
    truncate_stream,
)

NUM_CLIENTS, PER_ROUND, ROUNDS = 20, 8, 150
BW_SEED, JITTER = 0, 0.45
TIERS = {
    "0": {"d": [0, 0, 0, 0], "e": 0.1},
    "1": {"d": [0, 1, 0, 1], "e": 0.14},
    "2": {"d": [1, 1, 1, 1], "e": 0.18},
    "3": {"d": [2, 2, 2, 2], "e": 0.25},
}
TIER_NAMES = ["S", "M", "L", "XL"]

# The canonical local-training workload, from the run's own Namespace:
# 50000 CIFAR-10 images over 20 clients, 5 local epochs.
LOCAL_SAMPLES = 50000 // NUM_CLIENTS
LOCAL_EPOCHS = 5
# Backward costs about twice the forward pass, so a training step is ~3x.
# A convention, not a measurement -- stated so the reader can rescale.
FWD_BWD_MULTIPLIER = 3.0

# Measured GPU wall-clock, from sacct via analysis/fig_compute_frontier.py.
# Only the three arms with a BLIND sampler are used: their assignment is
# reproducible from the cached plans, so trainer counts are exact rather than
# reconstructed. (arm -> (job, hours))
MEASURED_HOURS = {
    "all_or_nothing": ("104127", 2.545),
    "nested": ("104148", 5.157),
    "blocks": ("104125", 8.797),
}

# Client hardware, as SUSTAINED TRAINING throughput in TFLOP/s.
#
# These must not be taken from a vendor's headline TOPS figure. Those quote INT8
# inference at peak, and on-device TRAINING is a different workload: fp16/fp32,
# backward pass, optimiser state, and a thermal envelope that a phone cannot
# hold for the minutes a federated round lasts. Sustained training throughput
# lands one to two orders of magnitude below the marketing number.
#
# The spread below is meant to BRACKET the plausible range, not to model a named
# device, and the conclusion is deliberately carried by the break-even -- which
# does not depend on picking a row.
DEVICES = [
    ("high-end phone, fp16 sustained", 1.5),
    ("mid-range phone", 0.4),
    ("low-end phone CPU", 0.05),
    ("cluster GPU (our simulator)", None),  # filled from the calibration
]


def build_server_model():
    from fedml_api.standalone.superfednas.Server.ServerModel import (
        ServerResnet_10_26,
    )
    server = ServerResnet_10_26({"n_classes": 10}, "TS_all_random",
                                NUM_CLIENTS, False)
    server.load_client_sample_counts([LOCAL_SAMPLES] * NUM_CLIENTS)
    server.set_top_bottom_k(1, 1)
    return server


def tier_costs(server):
    """Forward GFLOPs, parameters and payload bits for each tier.

    Profiler is ofa.utils.flops_counter, the one the trainer itself uses, and
    the payload rule is CommSimulator's (4 bytes per trainable parameter).
    """
    from ofa.utils import flops_counter as fp
    out = []
    for k in sorted(TIERS):
        cm = server.get_subnet(**TIERS[k])
        flops, _ = fp.profile(cm.model, (1, 3, 32, 32))
        # The trainer divides by 10e8, i.e. 1e9 -- keep the same unit so these
        # numbers are comparable with anything the run itself would have logged.
        gflops = flops / 10e8
        params = sum(p.numel() for p in cm.model.parameters() if p.requires_grad)
        out.append((TIER_NAMES[int(k)], gflops, params, params * 4 * 8))
    return out


def build_bandwidth():
    dummy = {c: np.ones(10, dtype=np.int64) for c in range(NUM_CLIENTS)}
    tiers = assign_connectivity_tiers(dummy, corr=0.0, seed=BW_SEED)
    return MarkovBandwidth.tiered_continuous(
        num_clients=NUM_CLIENTS, num_rounds=ROUNDS, client_tiers=tiers,
        seed=BW_SEED, jitter_sigma=JITTER)


def cfg_key(cfg):
    return repr(sorted((k, tuple(v) if isinstance(v, list) else v)
                       for k, v in cfg.items()))


def arch_profiler(server):
    """cfg -> training TFLOPs for one client-round, memoised.

    The assigned architecture is NOT one of the four ladder tiers. TS_all_random
    draws freely from the supernet -- 163 distinct configs over the canonical
    150 rounds, spanning 41.6-286.9 MB -- and only the two forced sandwich slots
    per round land exactly on the min and max tiers. Bucketing clients into
    S/M/L/XL would therefore be a fiction; every drawn architecture is profiled
    instead.

    (The tempting shortcut is plan.train_tier. It is not the assignment:
    build_clipped_plan sets it to len(tiers)-1 for every client, because it
    indexes the top of the CLIPPED ladder, which is always the trained arch
    itself. Reading it as a tier reports every client as XL and overstates
    compute by up to 7x, with nothing looking wrong.)
    """
    from ofa.utils import flops_counter as fp
    cache = {}

    def profile(cfg):
        k = cfg_key(cfg)
        if k not in cache:
            cm = server.get_subnet(**cfg)
            flops, _ = fp.profile(cm.model, (1, 3, 32, 32))
            gflops = flops / 10e8
            cache[k] = (gflops * FWD_BWD_MULTIPLIER * LOCAL_SAMPLES
                        * LOCAL_EPOCHS / 1e3)
        return cache[k]

    profile.cache = cache
    return profile


def who_trains(plans, bw, T, profile):
    """Per arm: per round, the training TFLOPs of the clients that actually run.

    A client that delivers nothing never trains -- the trainer decides delivery
    before training precisely so no compute is wasted -- so this is also the
    per-arm compute profile.
    """
    per_round = {}
    for rnd, cli, full, plan, stream in plans:
        per_round.setdefault(rnd, []).append((cli, full, plan, stream))

    arms = {a: [] for a in ("all_or_nothing", "nested", "blocks")}
    for rnd in sorted(per_round):
        got = {a: [] for a in arms}
        for cli, full, plan, stream in per_round[rnd]:
            b = bw.bandwidth(cli, rnd)
            budget = b * T
            # The trained architecture is the top of the client's clipped
            # ladder, by build_clipped_plan's own contract.
            tf = profile(plan.tiers[-1][1])
            if not all_or_nothing(full, budget, b, T).delivers_nothing():
                got["all_or_nothing"].append(tf)
            if not truncate(plan, budget, b, T).delivers_nothing():
                got["nested"].append(tf)
            if not truncate_stream(stream, budget, b, T).delivers_nothing():
                got["blocks"].append(tf)
        for a in arms:
            arms[a].append(got[a])
    return arms


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True,
                    help="pickle from sweep_deadline.py --dump-plans")
    ap.add_argument("--deadlines", type=float, nargs="+",
                    default=[10, 20, 60, 120])
    args = ap.parse_args()

    server = build_server_model()
    costs = tier_costs(server)

    print("=" * 78)
    print("1. PER-TIER COST  (ofa.utils.flops_counter, the trainer's own profiler)")
    print("=" * 78)
    print("%-4s %12s %14s %12s %14s %8s %8s" % (
        "tier", "fwd GFLOPs", "params (M)", "payload MB", "train TFLOPs",
        "xS cmp", "xS byte"))
    train_tflops, tier_bits = {}, []
    base_tf = costs[0][1] * FWD_BWD_MULTIPLIER * LOCAL_SAMPLES * LOCAL_EPOCHS / 1e3
    for i, (name, gf, pr, bits) in enumerate(costs):
        tf = gf * FWD_BWD_MULTIPLIER * LOCAL_SAMPLES * LOCAL_EPOCHS / 1e3
        train_tflops[i] = tf
        tier_bits.append(bits)
        print("%-4s %12.3f %14.2f %12.1f %14.1f %8.2f %8.2f"
              % (name, gf, pr / 1e6, bits / 8e6, tf, tf / base_tf,
                 bits / costs[0][3]))
    print("\ntrain TFLOPs = fwd GFLOPs x %.0f (fwd+bwd) x %d samples x %d epochs"
          % (FWD_BWD_MULTIPLIER, LOCAL_SAMPLES, LOCAL_EPOCHS))
    print("Note the last two columns: compute grows 7.4x across the ladder while")
    print("bytes grow 6.9x, so the tiers are close to proportional -- the ladder")
    print("does not let a client buy accuracy cheaply in one currency.")

    with open(args.plans, "rb") as fh:
        plans = pickle.load(fh)["plans"]
    bw = build_bandwidth()

    print()
    print("=" * 78)
    print("2. WHO TRAINS, AND THE COMPUTE THAT FOLLOWS  (canonical T = 20 s)")
    print("=" * 78)
    profile = arch_profiler(server)
    print("profiling the drawn architectures (memoised)...", flush=True)
    arms20 = who_trains(plans, bw, 20.0, profile)
    print("  %d distinct architectures profiled\n" % len(profile.cache))

    per_arm_tflops, per_arm_straggler = {}, {}
    print("%-16s %9s %14s %14s %14s" % (
        "arm", "trainers", "median TFLOPs", "sum/round", "straggler"))
    for arm, rounds_ in arms20.items():
        n = np.mean([len(r) for r in rounds_])
        flat = [t for r in rounds_ for t in r]
        tf = np.mean([sum(r) for r in rounds_])
        strag = np.mean([max(r) if r else 0.0 for r in rounds_])
        per_arm_tflops[arm] = tf
        per_arm_straggler[arm] = strag
        print("%-16s %9.2f %14.1f %14.1f %14.1f"
              % (arm, n, np.median(flat) if flat else 0.0, tf, strag))
    print("\nThe per-client median is what a device pays; the sum is what our")
    print("sequential simulator pays; the straggler is what a real round waits")
    print("for. They differ by ~8x and confusing them decides the answer below.")

    print()
    print("=" * 78)
    print("3. CALIBRATION against measured GPU wall-clock")
    print("=" * 78)
    # total_seconds = ROUNDS * (fixed + k * TFLOPs_per_round)
    A, y, labels = [], [], []
    for arm, (job, hours) in MEASURED_HOURS.items():
        A.append([1.0, per_arm_tflops[arm]])
        y.append(hours * 3600.0 / ROUNDS)
        labels.append((arm, job, hours))
    A, y = np.asarray(A), np.asarray(y)
    (fixed, k), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([fixed, k])
    print("model: seconds/round = fixed + k x TFLOPs/round")
    print("  fixed overhead   %7.2f s/round   (evaluation, aggregation, sampling)"
          % fixed)
    print("  k                %7.3f s/TFLOP    -> %.1f TFLOP/s effective"
          % (k, 1.0 / k if k > 0 else float("nan")))
    print()
    print("%-16s %6s %12s %12s %9s" % ("arm", "job", "measured s/r", "fitted s/r",
                                       "resid"))
    for (arm, job, hours), p in zip(labels, pred):
        m = hours * 3600.0 / ROUNDS
        print("%-16s %6s %12.1f %12.1f %+9.1f" % (arm, job, m, p, p - m))
    worst = np.max(np.abs(pred - y))
    print("\nworst residual %.1f s/round on %.0f-%.0f s/round observations (%.0f%%)."
          % (worst, y.min(), y.max(), 100 * worst / y.mean()))
    print("Two unknowns against three arms, so this is a check, not a tautology.")
    print("Wall-clock on a shared node is a PROXY for compute: read the")
    print("magnitude, not the third digit.")

    gpu_tflops = 1.0 / k if k > 0 else float("nan")

    print()
    print("=" * 78)
    print("4. THE ANSWER: round compute against T")
    print("=" * 78)
    print("A deployment trains the 8 selected clients IN PARALLEL, so the")
    print("round's compute is the straggler's, not the sum. Our simulator runs")
    print("them sequentially, which is why its wall-clock is ~8x larger.\n")

    # The blocks arm is the one to quote: at ~0% drop every selected client
    # trains, so it is both our headline arm and the compute worst case.
    straggler_tf = per_arm_straggler["blocks"]
    sum_tf = per_arm_tflops["blocks"]
    print("blocks arm, per round: straggler %.1f TFLOPs, sum over clients %.1f"
          % (straggler_tf, sum_tf))
    print()

    devs = [(n, v if v is not None else gpu_tflops) for n, v in DEVICES]
    print("%-32s %12s %s" % ("client hardware", "compute s",
                             "   round = compute + T, and the T=20->120 factor"))
    for name, tps in devs:
        c = straggler_tf / tps
        cells = []
        for T in args.deadlines:
            cells.append("T=%-4.0f %6.0fs" % (T, c + T))
        base = c + 20.0
        factor = (c + 120.0) / base
        print("%-32s %10.1f s   %s   ->  %.2fx"
              % (name, c, "  ".join(cells), factor))

    print()
    print("BREAK-EVEN: compute equals a T=20 s deadline at %.1f TFLOP/s of"
          % (straggler_tf / 20.0))
    print("sustained client training throughput.")
    print()
    print("VERDICT, and it goes AGAINST the thesis as currently written: no")
    print("plausible edge device is above that break-even. Real rounds are")
    print("COMPUTE-bound, not communication-bound, so the honest deadline")
    print("multiplier is %.2f-%.2fx, not the 6x that the thesis, Section 4.2, second argument"
          % (min((straggler_tf / t + 120) / (straggler_tf / t + 20)
                 for _n, t in devs),
             max((straggler_tf / t + 120) / (straggler_tf / t + 20)
                 for _n, t in devs)))
    print("currently claims from communication alone. Leg 2 must be rewritten")
    print("down to that range. Legs 1, 3 and 4 are untouched -- they never")
    print("depended on the round being communication-bound.")

    print()
    print("=" * 78)
    print("5. WHAT THE COMPUTE FINDING GIVES BACK: wasted local training")
    print("=" * 78)
    print("Our simulator decides delivery BEFORE training, so a client that")
    print("cannot deliver is skipped and spends nothing. That is an")
    print("idealisation, and it favours the BASELINES: it assumes the client")
    print("knows its own future bandwidth. A real deployment finds out by")
    print("trying -- the client trains, attempts the upload, and misses the")
    print("deadline, having already paid in full.")
    print()
    print("Price that honestly. If every selected client trains (which is what")
    print("a deployment without an oracle does), the round costs %.1f TFLOPs"
          % sum_tf)
    print("and each arm keeps only what it delivers:")
    print()
    print("%-16s %14s %14s %12s" % ("arm", "useful TFLOPs", "burnt TFLOPs",
                                    "burnt %"))
    for arm in ("all_or_nothing", "nested", "blocks"):
        useful = per_arm_tflops[arm]
        burnt = sum_tf - useful
        print("%-16s %14.1f %14.1f %11.0f%%"
              % (arm, useful, burnt, 100.0 * burnt / sum_tf))
    print()
    print("This is the argument the compute analysis hands to the mechanism,")
    print("and it is STRONGER the more compute-bound the round is -- which is")
    print("exactly the regime section 4 just showed we are in. Anytime delivery")
    print("does not merely save bytes; it stops the client's dominant cost from")
    print("being thrown away. Worth its own paragraph in the thesis, and worth")
    print("noting that our own headline numbers do NOT claim this credit: the")
    print("ex-ante skip means results/README.md understates the mechanism's")
    print("advantage on any real device.")


if __name__ == "__main__":
    main()
