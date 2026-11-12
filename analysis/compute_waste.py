#!/usr/bin/env python3
"""compute_waste -- the local compute MaxFit does not spend, and who stops spending it.

THE QUESTION. MaxFit's byte saving is measured (51.3 GB against 65.8, -22%).
Its COMPUTE saving is asserted everywhere and plotted nowhere, and it is the
more interesting half: a client that trains an architecture its link cannot
carry has burned battery for a gradient the server will only partly receive.

THE COMPARISON THAT IS HONEST, and why the obvious one is not. Under
``all_or_nothing`` the trainer SKIPS a client whose update cannot be delivered,
so that arm wastes no compute by construction -- it just does not learn from
those clients, which is the accuracy cost reported elsewhere. Comparing MaxFit
against it would show MaxFit spending MORE compute (more clients train at all)
and would say nothing about waste. The waste lives under CASCADE, where a client
trains its whole assigned architecture and then delivers only the prefix that
fits the deadline. So the comparison here is **blind CASCADE against MaxFit +
CASCADE**, two arms that deliver from every client and differ only in what each
client was asked to train.

WHAT IS MEASURED
  1. Training cost of every assigned architecture, from the run logs, priced by
     the SAME profiler the trainer uses (ofa.utils.flops_counter). The logged
     subnets are arbitrary points of the elastic space, not the four reference
     tiers, so their cost cannot be read off the ladder and must be computed.
  2. That cost split by the client's connectivity profile (poor / average /
     good), which is the mechanism: MaxFit should stop handing large
     architectures to clients that cannot ship them, and leave good clients
     alone.
  3. The totals over the run, at accuracy the two arms share to within the
     nondeterminism floor -- which is what makes a compute difference a saving
     rather than a trade.

WHAT IS NOT CLAIMED. This prices the compute ASSIGNED, not the compute whose
parameters arrived. Splitting the assigned cost into a delivered and an
undelivered part needs per-unit MAC accounting, which the block decomposition
supports but this script does not do. The assigned figure is the one the client
actually pays, so it is the right quantity for an energy argument regardless.

Runs on the cluster (needs torch for the profiler); writes a small CSV.

    python analysis/compute_waste.py --out analysis/csv/compute_waste.csv
"""

import argparse
import ast
import collections
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_code_root():
    for cand in (os.path.join(HERE, ".."), os.path.join(HERE, "..", "dynfednas_v2")):
        if os.path.isdir(os.path.join(cand, "fedml_api")):
            return os.path.abspath(cand)
    raise SystemExit("cannot locate fedml_api from %s" % HERE)


ROOT = _find_code_root()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(
    ROOT, "fedml_api", "standalone", "superfednas", "elastic_nn"))
# search_phase.py does `from test_superfednas_resnet import ...`, a top-level
# import that only resolves when its own directory is on the path -- it is
# normally run from there. Importing it as a package member does not put it
# there, so the directory goes on explicitly.
sys.path.insert(0, os.path.join(
    ROOT, "fedml_experiments", "standalone", "superfednas"))

# np.int64(0) / np.float64(0.18) survive into the logs; ast.literal_eval will not
# take them, and they carry no information a plain number does not.
_NPWRAP = re.compile(r"np\.(?:int64|float64)\(([^)]*)\)")
_ASSIGN = re.compile(
    r"Client_id: (\d+) Round Number: (\d+) Subnet: (\{.*?\})\s+LR:")
_TIERS = re.compile(r"connectivity tiers \([^)]*\): (\{.*\})")


def parse_log(path):
    """-> (profile_of_client, [(round, client, arch_dict)])."""
    profiles, rows = {}, []
    with open(path, errors="replace") as fh:
        for line in fh:
            if "connectivity tiers" in line and not profiles:
                m = _TIERS.search(line)
                if m:
                    for name, ids in ast.literal_eval(m.group(1)).items():
                        for c in ids:
                            profiles[c] = name
            m = _ASSIGN.search(line)
            if m:
                client, rnd, blob = int(m.group(1)), int(m.group(2)), m.group(3)
                arch = ast.literal_eval(_NPWRAP.sub(r"\1", blob))
                arch.pop("w", None)
                rows.append((rnd, client, arch))
    if not rows:
        raise SystemExit("no assignments parsed from %s" % path)
    return profiles, rows


def macs_pricer(ckpt_path):
    """-> f(arch) -> GMAC, memoized. Uses the trainer's own profiler."""
    import torch
    from search_phase import custom_ofa_net, get_flops  # noqa: E402
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # custom_ofa_net reads args.client_num_in_total off the training namespace;
    # nothing else on it is touched when only the profiler is wanted.
    shim = argparse.Namespace(client_num_in_total=20)
    model = custom_ofa_net(ckpt["supernet_class"], 10, ckpt["supernet_config"], shim)
    model.set_model_params(ckpt["params"])
    cache = {}

    def price(arch):
        key = (tuple(arch["d"]), tuple(arch["e"]))
        if key not in cache:
            cache[key] = get_flops(model, arch)[0] / 1e9
        return cache[key]

    price.cache = cache
    return price


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", required=True, nargs="+",
                    help="blind CASCADE run logs, one per seed")
    ap.add_argument("--maxfit", required=True, nargs="+",
                    help="MaxFit + CASCADE run logs, one per seed")
    ap.add_argument("--ckpt", required=True, help="any checkpoint of this supernet")
    ap.add_argument("--out", default=os.path.join(HERE, "csv", "compute_waste.csv"))
    args = ap.parse_args()

    price = macs_pricer(args.ckpt)
    out = []
    for arm, paths in (("blind", args.blind), ("maxfit", args.maxfit)):
        # One entry per seed per profile. The interval the figure draws is the
        # spread across SEEDS, which is what an error bar is read as, and not
        # the spread across client-rounds, which is a property of the sampler
        # rather than an uncertainty and belongs in the text.
        by_seed = collections.defaultdict(list)
        per_profile = collections.defaultdict(list)
        for path in paths:
            # each log carries its own connectivity split, which depends on the
            # bandwidth seed, so a seed is read with the mapping it ran under
            profiles, rows = parse_log(path)
            seed_prof = collections.defaultdict(list)
            for _, client, arch in rows:
                prof = profiles.get(client, "unknown")
                seed_prof[prof].append(price(arch))
                per_profile[prof].append(price(arch))
            for prof, vs in seed_prof.items():
                by_seed[prof].append(sum(vs) / len(vs))
            by_seed["all"].append(
                sum(v for vv in seed_prof.values() for v in vv)
                / sum(len(vv) for vv in seed_prof.values()))
        total = sum(v for vs in per_profile.values() for v in vs)
        n = sum(len(vs) for vs in per_profile.values())
        print("%-7s %d seeds, %d client-rounds, %.3f GMAC mean" % (
            arm, len(paths), n, total / n))
        # The five-number summary, not only the mean: a mean over 1200
        # client-rounds throws away the spread, and the spread is the point --
        # blind assignment gives every client the same DISTRIBUTION of
        # architectures, MaxFit gives each a different one.
        def summary(vs):
            v = sorted(vs)
            def q(p):
                k = (len(v) - 1) * p
                lo, hi = int(k), min(int(k) + 1, len(v) - 1)
                return v[lo] + (v[hi] - v[lo]) * (k - lo)
            q1, med, q3 = q(0.25), q(0.5), q(0.75)
            mu = sum(v) / len(v)
            sd = (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5
            iqr = q3 - q1
            lo = min(x for x in v if x >= q1 - 1.5 * iqr)
            hi = max(x for x in v if x <= q3 + 1.5 * iqr)
            return lo, q1, med, q3, hi, sd

        for prof in ("poor", "average", "good", "all"):
            vs = ([v for vv in per_profile.values() for v in vv] if prof == "all"
                  else per_profile.get(prof, []))
            if not vs:
                continue
            lo, q1, med, q3, hi, sd = summary(vs)
            seeds = by_seed.get(prof, [])
            seed_mean = sum(seeds) / len(seeds) if seeds else float("nan")
            seed_sd = ((sum((x - seed_mean) ** 2 for x in seeds)
                        / (len(seeds) - 1)) ** 0.5) if len(seeds) > 1 else 0.0
            print("        %-8s n=%-5d  seeds %d: %.3f +- %.3f  (median %.3f)"
                  % (prof, len(vs), len(seeds), seed_mean, seed_sd, med))
            out.append((arm, prof, len(vs), seed_mean, seed_sd,
                        lo, q1, med, q3, hi, sd))
    print("(%d distinct architectures priced)" % len(price.cache))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "profile", "client_rounds", "seed_mean", "seed_sd",
                    "whisker_lo", "q1", "median", "q3", "whisker_hi", "roundwise_sd"])
        w.writerows(out)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
