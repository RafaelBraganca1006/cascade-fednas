#!/usr/bin/env python3
"""fetch_mahimahi -- the Sprout uplink traces, as a compact archive.

WHY THESE EXIST IN THIS PROJECT. Every other public cellular trace set we could
find measures DOWNLINK, because they were built by the video-streaming
community. Our deadline is an uplink deadline, so the Irish archive is replayed
under a documented substitution: downlink stands in for uplink, which is
optimistic and therefore safe for our claim, but it is a substitution.

These are not. From the Mahimahi trace README:

    "These trace files represent the time-varying CAPACITY of U.S. cellular
     networks as experienced by a mobile user. They were recorded using the
     Saturator tool."

    K. Winstein, A. Sivaraman, H. Balakrishnan. "Stochastic Forecasts Achieve
    High Throughput and Low Delay over Cellular Networks." USENIX NSDI 2013.

The Saturator drove the uplink and the downlink to saturation at the same time
and logged when each packet crossed, so an uplink file is a measurement of what
the uplink could carry, not of what an application happened to send. There is no
ACK-traffic trap here and no application limit: the two failure modes that ruled
out the Irish UL_bitrate column and the 5G dataset entirely.

FORMAT. One integer per line, a millisecond timestamp at which one 1500-byte
packet could cross. A repeated timestamp means several packets in that
millisecond. Capacity over any window is therefore
(packets in window) * 1500 * 8 / window_seconds.

WHAT THEY COST US, so it is on the record. They are 2013 LTE, UMTS and EV-DO
(plus one AT&T LTE run from 2016); there are eight of them against the Irish
archive's 135; they run about eight minutes each against fifteen; and they are
drive tests, so there is no static, pedestrian, bus or train split. They are the
right measurement of the wrong year, and the Irish archive is the wrong
measurement of the right variety, which is why this project keeps both.

    python analysis/fetch_mahimahi.py
"""

import gzip
import io
import os
import sys
import urllib.request

import numpy as np

RAW = "https://raw.githubusercontent.com/ravinet/mahimahi/master/traces/"
NAMES = [
    "ATT-LTE-driving.up",
    "ATT-LTE-driving-2016.up",
    "TMobile-LTE-driving.up",
    "TMobile-LTE-short.up",
    "TMobile-UMTS-driving.up",
    "Verizon-EVDO-driving.up",
    "Verizon-LTE-driving.up",
    "Verizon-LTE-short.up",
]
MTU_BITS = 1500 * 8
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "data", "sprout_uplink.npz")


def per_second_kbps(timestamps_ms):
    """Packet-delivery opportunities -> capacity in kbit/s, one value a second.

    A second with no opportunity is a real zero here, unlike the Irish archive's
    idle seconds: the Saturator always had data to send, so nothing crossing
    means nothing could cross.
    """
    if len(timestamps_ms) == 0:
        return np.zeros(0, dtype=np.int32)
    secs = timestamps_ms // 1000
    n = int(secs[-1]) + 1
    counts = np.bincount(secs, minlength=n)
    return (counts * MTU_BITS / 1000.0).astype(np.int32)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    os.makedirs(os.path.dirname(out), exist_ok=True)

    series, names = [], []
    for name in NAMES:
        url = RAW + name
        print("fetching %s ..." % name)
        with urllib.request.urlopen(url, timeout=300) as r:
            raw = r.read().decode("ascii", errors="replace")
        ts = np.array([int(x) for x in raw.split() if x.strip()], dtype=np.int64)
        kbps = per_second_kbps(ts)
        series.append(kbps)
        names.append(name[:-3])
        print("    %6.1f s   mean %6.2f Mbit/s   median %6.2f   p10 %5.2f   zeros %.1f%%"
              % (len(kbps), kbps.mean() / 1000, np.median(kbps) / 1000,
                 np.percentile(kbps, 10) / 1000, 100.0 * (kbps == 0).mean()))

    offsets = np.cumsum([0] + [len(s) for s in series]).astype(np.int64)
    np.savez_compressed(out, kbps=np.concatenate(series), offsets=offsets,
                        names=np.array(names))
    print("\nwrote %s (%.2f MB, %d traces)"
          % (out, os.path.getsize(out) / 1e6, len(series)))


if __name__ == "__main__":
    main()
