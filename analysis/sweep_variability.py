#!/usr/bin/env python3
"""sweep_variability -- delivery against the synthetic channel's within-state
spread, at a fixed deadline and a fixed mean throughput.

WHY THIS EXISTS
===============
The advisor's question was whether CASCADE's advantage grows with communication
variability, and no plot varied variability. The deadline sweep varies pressure
(payload against budget rho * T), which is a different axis; the three channels
differ in level as well as spread, so they are not a clean ordering either.

TWO KNOBS, AND WHY THE DEFAULT IS NOT THE OBVIOUS ONE
    jitter    the paper's ``s``: log-normal spread inside a Markov state. It is
              mean-normalised, so it adds variance without adding bandwidth, but
              it is a WEAK knob: the std of log throughput only moves from 1.32
              (s=0) to 1.61 (s=1.0), because almost all of the variation is the
              jump between the 50 / 10 / 2 Mbit/s states, not the spread inside
              one. A plot over it would be nearly flat whatever the mechanism.
    contrast  (default) the log-distance of the three states from their
              geometric mean is scaled by lambda: lambda = 1 is the canonical
              50 / 10 / 2, lambda = 0 is one constant rate, lambda > 1 is a wider
              gap. This moves the std of log throughput over a wide range
              while the Markov dynamics, the profiles and the jitter stay fixed.
In both cases the chain draws and the jitter draws use separate RNG streams, so
every setting replays the SAME state trajectories.

TWO THINGS THE SWEEP DOES TO KEEP THE COMPARISON HONEST
    1. Every setting is rescaled so that the global mean throughput equals the
       canonical one (``--no-renorm`` turns it off). Otherwise a rise in spread
       would also be a fall (or rise) in mean bandwidth, which is the confound the
       jitter's own normalisation exists to remove. The jitter is also clipped at
       +-2 sigma AFTER that normalisation, so its mean drifts at large s.
    2. The realized spread is REPORTED, not assumed: the std of log throughput
       and the mean, per row. The x axis of a plot should be the realized std of
       log throughput, comparable to the 1.38 (synthetic) and 1.15 (measured)
       figures the paper quotes, not the knob.

It is offline for the same reason sweep_deadline.py is: drop, waste and volume
are decided by (payload, bandwidth, deadline) before any gradient exists. It
reuses that script's ``sweep()``, which calls the production transmission code.

HOW TO RUN
    1. ONCE, where torch and ofa are installed (the cluster; CPU is enough, no
       GPU quota is spent):
           python analysis/sweep_deadline.py --dump-plans analysis/plans.pkl
    2. Copy plans.pkl to the Mac and run everything else there (numpy only):
           python analysis/sweep_variability.py --plans analysis/plans.pkl \\
               --out analysis/csv/variability_sweep.csv
"""

import argparse
import csv
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_deadline as sd  # noqa: E402  (installs the code root on sys.path)

from fedml_api.standalone.superfednas.comm_sim.bandwidth import (  # noqa: E402
    MarkovBandwidth,
    assign_connectivity_tiers,
)

# The three methods the question is about. The fixed-payload baselines are
# reported too, since FedAvg is the other end of the all-or-nothing family.
KEEP = ("SuperFedNAS (all-or-nothing)", "DynFedNAS (block stream)",
        "DynFedNAS (nested)", "FedAvg", "FedNAS-DARTS")


def build_chain(value, bw_seed, knob="contrast"):
    """The canonical chain with one variability knob moved. Same tiers, same
    state trajectories for every value; ``value`` is lambda for ``contrast`` and
    the within-state spread s for ``jitter``."""
    dummy = {c: np.ones(10, dtype=np.int64) for c in range(sd.NUM_CLIENTS)}
    tiers = assign_connectivity_tiers(dummy, corr=0.0, seed=bw_seed)
    states = np.array([50.0, 10.0, 2.0]) * 1e6
    jitter = sd.JITTER
    if knob == "contrast":
        mu = np.log(states).mean()
        states = np.exp(mu + value * (np.log(states) - mu))
    else:
        jitter = value
    per_client = [MarkovBandwidth.TIER_TRANSITIONS[t] for t in tiers]
    return MarkovBandwidth(
        sd.NUM_CLIENTS, sd.ROUNDS, states,
        MarkovBandwidth.TIER_TRANSITIONS[1], seed=bw_seed,
        per_client_transition=per_client, client_tiers=list(tiers),
        jitter_sigma=jitter)


def realized(chain):
    """Mean throughput (Mbit/s) and std of log throughput over every
    (client, round) the replay will query."""
    bw = chain._bw
    return float(bw.mean() / 1e6), float(np.log(bw).std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True,
                    help="pickle written by sweep_deadline.py --dump-plans")
    ap.add_argument("--knob", choices=["contrast", "jitter"],
                    default="contrast")
    ap.add_argument("--spreads", type=float, nargs="+", default=None,
                    help="knob values. contrast: lambda, default 0 0.25 0.5 "
                         "0.75 1 1.25 (1 = canonical). jitter: s, default "
                         "0 0.25 0.45 0.7 1 (0.45 = canonical)")
    ap.add_argument("--no-renorm", action="store_true")
    ap.add_argument("--deadlines", type=float, nargs="+", default=[20.0])
    ap.add_argument("--bw-seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="chain realizations; one is exact but gives no band")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "csv",
        "variability_sweep.csv"))
    args = ap.parse_args()
    canonical = 1.0 if args.knob == "contrast" else sd.JITTER
    if args.spreads is None:
        args.spreads = ([0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
                        if args.knob == "contrast"
                        else [0.0, 0.25, 0.45, 0.7, 1.0])

    with open(args.plans, "rb") as fh:
        blob = pickle.load(fh)
    plans, fixed = blob["plans"], blob["fixed"]
    if blob["rounds"] != sd.ROUNDS:
        sys.exit("the pickle holds %d rounds, the sweep needs %d"
                 % (blob["rounds"], sd.ROUNDS))
    print("loaded %d plan slots" % len(plans))

    rows = []
    for seed in args.bw_seeds:
        ref_mean, _ = realized(build_chain(canonical, seed, args.knob))
        for s in args.spreads:
            chain = build_chain(s, seed, args.knob)
            if not args.no_renorm:
                chain._bw *= (ref_mean * 1e6) / chain._bw.mean()
            mean_mbps, std_log = realized(chain)
            for r in sd.sweep(plans, lambda T: chain, args.deadlines, fixed):
                if r["method"] not in KEEP:
                    continue
                r.update(spread_s=s, knob=args.knob, bw_seed=seed, mean_mbps=mean_mbps,
                         std_log_bw=std_log)
                rows.append(r)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "method", "knob", "spread_s", "bw_seed", "deadline_s", "mean_mbps",
            "std_log_bw", "drop_pct", "waste_pct", "attempted_gb",
            "wall_clock_h"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["method"], r["deadline_s"],
                                             r["bw_seed"], r["spread_s"])):
            w.writerow({k: r[k] for k in w.fieldnames})
    print("CSV -> %s" % args.out)

    # Mean over seeds, one line per (method, deadline, s): what a plot reads.
    print("\n%-30s %5s %5s %8s %8s %8s %8s" % (
        "method", "T", "s", "std_log", "mean", "drop%", "waste%"))
    keys = sorted({(r["method"], r["deadline_s"], r["spread_s"]) for r in rows})
    for m, T, s in keys:
        sel = [r for r in rows if (r["method"], r["deadline_s"],
                                   r["spread_s"]) == (m, T, s)]
        print("%-30s %5g %5.2f %8.2f %8.2f %8.1f %8.1f" % (
            m, T, s, np.mean([r["std_log_bw"] for r in sel]),
            np.mean([r["mean_mbps"] for r in sel]),
            np.mean([r["drop_pct"] for r in sel]),
            np.mean([r["waste_pct"] for r in sel])))

    # Self-check: at the canonical setting and seed 0 the numbers must match the
    # deadline sweep the paper already reports.
    ref = [r for r in rows if r["bw_seed"] == 0 and r["deadline_s"] == 20.0
           and abs(r["spread_s"] - canonical) < 1e-9]
    for r in ref:
        if r["method"] in sd.MEASURED_T20:
            md, mw, job = sd.MEASURED_T20[r["method"]]
            print("check %-30s drop %.1f (measured %.1f)  waste %.1f (%.1f)"
                  % (r["method"], r["drop_pct"], md, r["waste_pct"], mw))


if __name__ == "__main__":
    main()
