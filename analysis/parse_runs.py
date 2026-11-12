#!/usr/bin/env python3
"""Parse dynFedNAS training .out logs into comparison tables and CSVs.

Usage:
    python3 analysis/parse_runs.py results/*.out
    python3 analysis/parse_runs.py --csv-dir analysis/csv results/v2_arm_*.out

Extracts per run:
  - config (deadline, tx_mode, sampler, rounds, partition alpha, bw seed)
  - per-round comm series (participation, drops, max-subnet fail, MB sent)
  - eval checkpoints: test accuracy per tier (S/M/L/XL by expand ratio)
  - v2 mechanism counters (truncated deliveries, carrier deliveries)
  - final summary line (drop rate, max failure rate, wall clock, MB totals)

Prints a cross-run comparison table (the three-arm view) and writes one CSV
per run (round-level) plus a combined summary CSV. Stdlib only — matplotlib
is not required (plotting happens elsewhere / dashboard).
"""

import argparse
import ast
import csv
import os
import re
import sys

TIER_BY_E = {0.1: "S", 0.14: "M", 0.18: "L", 0.25: "XL"}

RE_COMM = re.compile(
    r"\[comm_sim\] round (\d+): participation=([\d.]+) dropped=(\d+)/(\d+) "
    r"max_subnet_present=(\d) max_subnet_failed=(\d) MB_sent=([\d.]+)"
    # cumulative tail (logging upgrade 2026-07-15; absent in older logs)
    r"(?: MB_delivered_total=([\d.]+) MB_attempted_total=([\d.]+) "
    r"MB_wasted_total=([\d.]+) wall_clock_s=([\d.]+))?"
)
RE_TEST_HDR = re.compile(r"#+local_test_on_all_clients : (\d+)")
RE_METRIC = re.compile(r"INFO:root:(\{'(?:test|train)_acc'.*\})")
RE_SUMMARY = re.compile(
    r"\[comm_sim\] run summary: drop_rate=([\d.]+) max_subnet_failure_rate=([\d.]+) "
    r"wall_clock=([\d.]+)s total_MB_sent=([\d.]+) total_MB_dropped=([\d.]+)"
    r"(?: total_MB_attempted=([\d.]+) total_MB_wasted=([\d.]+) waste_rate=([\d.]+))?"
)
RE_NAMESPACE = re.compile(r"INFO:root:Namespace\((.*)\)\s*$")


def parse_namespace(line):
    """Pull the config fields we care about out of the argparse Namespace repr."""
    fields = {}
    for key in (
        "bw_deadline", "tx_mode", "subnet_dist_type", "comm_round",
        "partition_alpha", "bw_seed", "bw_estimate", "shard_carriers",
        "wandb_run_name", "init_seed",
    ):
        m = re.search(r"%s=('[^']*'|[^,)]+)" % key, line)
        if m:
            fields[key] = m.group(1).strip("'")
    return fields


def tier_of(subnet):
    e = subnet.get("e")
    if isinstance(e, (list, tuple)):
        e = e[0] if len(set(e)) == 1 else None
    return TIER_BY_E.get(e)


def parse_run(path):
    run = {
        "file": os.path.basename(path),
        "config": {},
        "rounds": [],          # per-round comm dicts
        "evals": {},           # round -> {tier: test_acc}
        "truncated": 0,
        "carrier_deliveries": 0,
        "summary": {},
    }
    cur_eval_round = None
    with open(path, errors="replace") as f:
        for line in f:
            if not run["config"]:
                m = RE_NAMESPACE.search(line)
                if m:
                    run["config"] = parse_namespace(m.group(1))
                    continue
            m = RE_COMM.search(line)
            if m:
                r = {
                    "round": int(m.group(1)),
                    "participation": float(m.group(2)),
                    "dropped": int(m.group(3)),
                    "selected": int(m.group(4)),
                    "max_present": int(m.group(5)),
                    "max_failed": int(m.group(6)),
                    "MB_sent": float(m.group(7)),
                }
                if m.group(8) is not None:
                    r["MB_delivered_cum"] = float(m.group(8))
                    r["MB_attempted_cum"] = float(m.group(9))
                    r["MB_wasted_cum"] = float(m.group(10))
                    r["wall_clock_s"] = float(m.group(11))
                run["rounds"].append(r)
                continue
            m = RE_TEST_HDR.search(line)
            if m:
                cur_eval_round = int(m.group(1))
                run["evals"].setdefault(cur_eval_round, {})
                continue
            if cur_eval_round is not None and "'test_acc'" in line:
                m = RE_METRIC.search(line)
                if m:
                    d = ast.literal_eval(m.group(1))
                    tier = tier_of(d.get("subnet", {}))
                    if tier:
                        run["evals"][cur_eval_round][tier] = d["test_acc"]
                continue
            if "truncated: trained" in line:
                run["truncated"] += 1
                continue
            if re.search(r"carrier \d+ delivered", line):
                run["carrier_deliveries"] += 1
                continue
            m = RE_SUMMARY.search(line)
            if m:
                run["summary"] = {
                    "drop_rate": float(m.group(1)),
                    "max_fail_rate": float(m.group(2)),
                    "wall_clock_s": float(m.group(3)),
                    "MB_sent": float(m.group(4)),
                    "MB_dropped": float(m.group(5)),
                }
                if m.group(6) is not None:
                    run["summary"]["MB_attempted"] = float(m.group(6))
                    run["summary"]["MB_wasted"] = float(m.group(7))
                    run["summary"]["waste_rate"] = float(m.group(8))
    return run


def label(run):
    cfg = run["config"]
    name = cfg.get("wandb_run_name") or run["file"]
    return name


def final_evals(run):
    if not run["evals"]:
        return {}, None
    last_round = max(run["evals"])
    return run["evals"][last_round], last_round


def print_table(runs):
    cols = [
        "run", "tx_mode", "T", "rounds", "drop%", "maxfail%",
        "S", "M", "L", "XL", "spread", "trunc", "carrier",
    ]
    print("\t".join(cols))
    for run in runs:
        cfg, summ = run["config"], run["summary"]
        accs, last = final_evals(run)
        spread = (
            "%.1f" % (100 * (accs["XL"] - accs["S"]))
            if "XL" in accs and "S" in accs else "-"
        )
        row = [
            label(run),
            cfg.get("tx_mode", "-"),
            cfg.get("bw_deadline", "-"),
            "%d@%s" % (len(run["rounds"]), ("eval@%s" % last) if last is not None else "-"),
            "%.1f" % (100 * summ["drop_rate"]) if summ else "-",
            "%.1f" % (100 * summ["max_fail_rate"]) if summ else "-",
        ]
        for tier in ("S", "M", "L", "XL"):
            row.append("%.2f" % (100 * accs[tier]) if tier in accs else "-")
        row += [spread, str(run["truncated"]), str(run["carrier_deliveries"])]
        print("\t".join(row))


def write_csvs(runs, csv_dir):
    os.makedirs(csv_dir, exist_ok=True)
    for run in runs:
        base = os.path.splitext(run["file"])[0]
        with open(os.path.join(csv_dir, base + "_rounds.csv"), "w", newline="") as f:
            fields = ["round", "participation", "dropped", "selected",
                      "max_present", "max_failed", "MB_sent",
                      "MB_delivered_cum", "MB_attempted_cum", "MB_wasted_cum",
                      "wall_clock_s", "acc_S", "acc_M", "acc_L", "acc_XL"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in run["rounds"]:
                row = dict(r)
                for tier, acc in run["evals"].get(r["round"], {}).items():
                    row["acc_" + tier] = acc
                w.writerow(row)
    with open(os.path.join(csv_dir, "summary.csv"), "w", newline="") as f:
        fields = ["run", "file", "tx_mode", "bw_deadline", "subnet_dist_type",
                  "partition_alpha", "rounds_logged", "drop_rate",
                  "max_fail_rate", "MB_sent", "MB_dropped", "MB_attempted",
                  "MB_wasted", "waste_rate", "truncated",
                  "carrier_deliveries", "acc_S", "acc_M", "acc_L", "acc_XL"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for run in runs:
            accs, _ = final_evals(run)
            row = {
                "run": label(run),
                "file": run["file"],
                "tx_mode": run["config"].get("tx_mode"),
                "bw_deadline": run["config"].get("bw_deadline"),
                "subnet_dist_type": run["config"].get("subnet_dist_type"),
                "partition_alpha": run["config"].get("partition_alpha"),
                "rounds_logged": len(run["rounds"]),
                "truncated": run["truncated"],
                "carrier_deliveries": run["carrier_deliveries"],
            }
            row.update({k: v for k, v in run["summary"].items()
                        if k in ("drop_rate", "max_fail_rate", "MB_sent", "MB_dropped",
                                 "MB_attempted", "MB_wasted", "waste_rate")})
            for tier, acc in accs.items():
                row["acc_" + tier] = acc
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outs", nargs="+", help=".out files to parse")
    ap.add_argument("--csv-dir", default=None, help="also write per-run + summary CSVs here")
    args = ap.parse_args()

    runs = [parse_run(p) for p in args.outs]
    runs = [r for r in runs if r["rounds"] or r["evals"]]
    if not runs:
        print("no parsable runs found", file=sys.stderr)
        sys.exit(1)
    runs.sort(key=lambda r: (str(r["config"].get("bw_deadline")), str(r["config"].get("tx_mode"))))
    print_table(runs)
    if args.csv_dir:
        write_csvs(runs, args.csv_dir)
        print("\nCSVs written to %s/" % args.csv_dir, file=sys.stderr)


if __name__ == "__main__":
    main()
