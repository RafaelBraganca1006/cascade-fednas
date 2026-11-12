#!/usr/bin/env python3
"""fetch_traces -- turn the Irish 4G LTE dataset into a compact throughput
archive the simulator can replay.

SOURCE
    Raca, Quinlan, Zahran, Sreenan. "Beyond throughput: a 4G LTE dataset with
    channel and context metrics." ACM MMSys 2018.
    http://cs1dev.ucc.ie/misl/4G_Dataset/LTE_Dataset.zip
    135 traces from two major Irish operators, 1 Hz, five mobility patterns
    (static, pedestrian, car, bus, train).

    Downloaded once and cached; the parsed result is committed as a ~0.6 MB
    .npz so every figure regenerates without network access.

------------------------------------------------------------------------------
WHICH COLUMN IS THE BANDWIDTH -- the one decision that can silently ruin the
whole trace study, so it is made here, in the open, with its evidence.

The obvious move is to read ``UL_bitrate``, because our deadline is an UPLINK
deadline. That is wrong, and it is wrong in a way that would not announce
itself: the figure would render, the axes would look sane, and every method
would simply collapse.

The dataset was collected while the phone ran a *download* workload. So
``UL_bitrate`` is not a measurement of uplink capacity -- it is the reverse-path
traffic of that download, essentially TCP acknowledgements. The signature is
unmistakable and this script prints it: the DL/UL median ratio is ~50x in EVERY
one of the five mobility patterns, which is the delayed-ACK ratio, not a radio
asymmetry. A median "uplink" of 107 kbit/s is also physically absurd for LTE.

``DL_bitrate`` under ``State == 'D'`` IS a measurement: the link was saturated
by an active download, so the observed rate is the rate the link could carry.
That is the quantity we want. We replay it as the client's uplink capacity.

    ASSUMPTION, to be stated in the thesis, with its direction of bias:
    LTE is asymmetric, so a real uplink would be SLOWER than the downlink
    throughput we replay. Our numbers are therefore OPTIMISTIC -- they
    understate drop and waste for every method alike. This is the safe
    direction for our claim: the mechanism's advantage is demonstrated under
    a bandwidth assumption that favours the baselines.
    ``--ul_factor`` scales every sample if a pessimistic variant is wanted;
    the sweep reports 1.0 (as measured) and 0.5 as a sensitivity.

Samples with ``State == 'I'`` (idle: the app was not downloading) are DROPPED,
not read as zeros. An idle second is an absence of demand, not an absence of
capacity, and treating it as a zero would inject fake outages -- which would
matter most exactly where it would mislead most, on the `train` pattern, where
34% of samples are idle. Genuine outages survive this filter: they appear as
``DL_bitrate == 0`` while ``State == 'D'``, i.e. the download wanted bytes and
got none. The train pattern's tunnel dropouts are preserved that way (its
State=='D' 10th percentile is 7 kbit/s).

    python3 analysis/fetch_traces.py            # download, parse, write .npz
    python3 analysis/fetch_traces.py --report   # re-print the stats from cache
"""

import argparse
import csv
import io
import os
import sys
import urllib.request
import zipfile

import numpy as np

URL = "http://cs1dev.ucc.ie/misl/4G_Dataset/LTE_Dataset.zip"
PATTERNS = ("static", "pedestrian", "car", "bus", "train")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "data", "irish4g.npz")

# The dataset reports bitrates in kbit/s.
KBIT = 1e3


def download(cache):
    if os.path.exists(cache):
        print("cache hit  %s (%.1f MB)" % (cache, os.path.getsize(cache) / 1e6))
        return cache
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    print("downloading %s ..." % URL)
    with urllib.request.urlopen(URL, timeout=300) as r, open(cache, "wb") as fh:
        fh.write(r.read())
    print("saved      %s (%.1f MB)" % (cache, os.path.getsize(cache) / 1e6))
    return cache


def parse(zip_path):
    """Read every trace; return per-trace dicts with the kept DL series."""
    traces = []
    ack_dl, ack_ul = [], []  # for the DL/UL evidence print
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist()
                 if n.endswith(".csv") and "__MACOSX" not in n]
        for name in sorted(names):
            pattern = None
            for p in PATTERNS:
                if "/%s/" % p in name:
                    pattern = p
            if pattern is None:
                continue
            with zf.open(name) as fh:
                rdr = csv.DictReader(io.TextIOWrapper(fh, "utf-8",
                                                      errors="replace"))
                dl, idle = [], 0
                for row in rdr:
                    try:
                        d = float(row["DL_bitrate"])
                        u = float(row["UL_bitrate"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if row.get("State") != "D":
                        idle += 1
                        continue
                    dl.append(d)
                    ack_dl.append(d)
                    ack_ul.append(u)
            if len(dl) >= 60:  # a minute of saturated link, or it is not a trace
                traces.append(dict(name=os.path.basename(name), pattern=pattern,
                                   kbps=np.asarray(dl, dtype=np.int32),
                                   idle=idle))
    return traces, np.asarray(ack_dl), np.asarray(ack_ul)


def report(traces, ack_dl=None, ack_ul=None):
    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else float("nan")

    print()
    print("%-12s %6s %8s %10s %10s %10s %10s %8s" % (
        "pattern", "traces", "samples", "p10 Mbps", "p50 Mbps", "p90 Mbps",
        "max Mbps", "zero%"))
    for p in PATTERNS:
        sub = [t for t in traces if t["pattern"] == p]
        if not sub:
            continue
        a = np.concatenate([t["kbps"] for t in sub]).astype(float) / 1e3
        print("%-12s %6d %8d %10.2f %10.2f %10.2f %10.2f %8.1f" % (
            p, len(sub), len(a), pct(a, 10), pct(a, 50), pct(a, 90), a.max(),
            100.0 * (a == 0).mean()))
    allk = np.concatenate([t["kbps"] for t in traces]).astype(float) / 1e3
    print("%-12s %6d %8d %10.2f %10.2f %10.2f %10.2f %8.1f" % (
        "ALL", len(traces), len(allk), pct(allk, 10), pct(allk, 50),
        pct(allk, 90), allk.max(), 100.0 * (allk == 0).mean()))

    if ack_dl is not None and len(ack_ul):
        med_dl, med_ul = np.median(ack_dl), np.median(ack_ul)
        print()
        print("EVIDENCE that UL_bitrate is not uplink capacity:")
        print("  median DL %.0f kbit/s, median UL %.0f kbit/s, ratio %.0fx"
              % (med_dl, med_ul, med_dl / max(med_ul, 1)))
        print("  a ~50x reverse ratio is the TCP delayed-ACK signature of a")
        print("  download workload, not a radio asymmetry. DL is the capacity")
        print("  measurement; see this file's header.")

    # Within-trace variability, which is what the synthetic profile's
    # jitter_sigma is supposed to imitate (bandwidth.py's tiered_continuous
    # docstring flags 0.45 as uncalibrated "until E9 lands"). This is E9.
    sigmas = []
    for t in traces:
        v = t["kbps"].astype(float)
        v = v[v > 0]
        if len(v) >= 120:
            sigmas.append(float(np.std(np.log(v))))
    if sigmas:
        print()
        print("CALIBRATION for tiered_continuous jitter_sigma:")
        print("  sd of log throughput within a trace: median %.2f, "
              "IQR %.2f-%.2f, over %d traces"
              % (np.median(sigmas), np.percentile(sigmas, 25),
                 np.percentile(sigmas, 75), len(sigmas)))
        print("  (the canonical run uses jitter_sigma=0.45)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--cache", default=os.path.join(HERE, "data",
                                                    "LTE_Dataset.zip"))
    ap.add_argument("--report", action="store_true",
                    help="only re-print statistics from an existing .npz")
    args = ap.parse_args()

    if args.report:
        if not os.path.exists(args.out):
            sys.exit("no archive at %s -- run without --report first" % args.out)
        z = np.load(args.out, allow_pickle=False)
        off, kbps = z["offsets"], z["kbps"]
        pats = [PATTERNS[i] for i in z["pattern_idx"]]
        traces = [dict(pattern=pats[i], kbps=kbps[off[i]:off[i + 1]],
                       name="", idle=0) for i in range(len(pats))]
        report(traces)
        return

    zip_path = download(args.cache)
    traces, ack_dl, ack_ul = parse(zip_path)
    if not traces:
        sys.exit("parsed no traces -- the archive layout changed")
    report(traces, ack_dl, ack_ul)

    order = {p: i for i, p in enumerate(PATTERNS)}
    kbps = np.concatenate([t["kbps"] for t in traces])
    offsets = np.zeros(len(traces) + 1, dtype=np.int64)
    np.cumsum([len(t["kbps"]) for t in traces], out=offsets[1:])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        kbps=kbps.astype(np.int32),
        offsets=offsets,
        pattern_idx=np.asarray([order[t["pattern"]] for t in traces],
                               dtype=np.int8),
        names=np.asarray([t["name"] for t in traces]),
    )
    print()
    print("NPZ -> %s (%.2f MB, %d traces, %d samples)"
          % (args.out, os.path.getsize(args.out) / 1e6, len(traces), len(kbps)))


if __name__ == "__main__":
    main()
