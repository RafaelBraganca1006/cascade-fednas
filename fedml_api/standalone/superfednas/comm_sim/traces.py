"""Loading measured bandwidth traces for :class:`TraceReplayBandwidth`.

The archive is produced by ``analysis/fetch_traces.py`` from the Irish 4G LTE
dataset (Raca et al., MMSys 2018) and committed to the repository, so nothing
here touches the network. Read that script's header for the two decisions that
shape the data: throughput is taken from ``DL_bitrate`` under ``State == 'D'``
(``UL_bitrate`` is TCP acknowledgement traffic, not uplink capacity), and idle
samples are dropped rather than read as zeros.

This module only maps traces onto clients. It is deliberately separate from
``bandwidth.py`` so the model stays free of file paths and archive layout.
"""

import os

import numpy as np

PATTERNS = ("static", "pedestrian", "car", "bus", "train")

ARCHIVE_NAME = os.path.join("analysis", "data", "irish4g.npz")

# The archive is looked up relative to this file, at two depths: `analysis/` is
# a sibling of `fedml_api` in the cluster checkout, and one level further up in
# the full working tree. Probing beats hard-coding one of them, because the
# wrong guess would surface as a FileNotFoundError on the cluster only.
_CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          *([".."] * 4)))
_CANDIDATES = (os.path.join(_CODE_ROOT, ARCHIVE_NAME),
               os.path.join(os.path.dirname(_CODE_ROOT), ARCHIVE_NAME))


def archive_path(path=None):
    if path:
        return os.path.abspath(path)
    for c in _CANDIDATES:
        if os.path.exists(c):
            return c
    return _CANDIDATES[-1]  # the one whose absence the error message names


def load_archive(path=None):
    """Return ``(list[np.ndarray] of bits/sec, list[str] pattern)``."""
    p = archive_path(path)
    if not os.path.exists(p):
        raise FileNotFoundError(
            "no trace archive at %s -- run `python3 analysis/fetch_traces.py` "
            "to build it from the published dataset" % p)
    z = np.load(p, allow_pickle=False)
    kbps, offsets, pat = z["kbps"], z["offsets"], z["pattern_idx"]
    traces = [kbps[offsets[i]:offsets[i + 1]].astype(float) * 1e3
              for i in range(len(pat))]
    return traces, [PATTERNS[i] for i in pat]


def load_client_traces(num_clients, seed=0, patterns=None, ul_factor=1.0,
                       path=None, min_samples=120):
    """Assign one measured trace to each client.

    Selection is *without replacement* whenever there are enough traces, so a
    20-client population is 20 genuinely different measured links rather than
    one link repeated -- the heterogeneity is the dataset's, not an artefact of
    resampling. With fewer traces than clients it falls back to replacement and
    the differing start offsets keep the clients from moving in lockstep.

    ``patterns`` restricts to a subset of mobility patterns (e.g. ``["train"]``
    for the outage-heavy stress case); ``None`` uses all five, which is the
    heterogeneous default.

    ``ul_factor`` scales every sample. It exists for one honest reason: the
    archive records downlink throughput, and a real uplink is slower. 1.0 is
    "as measured" (optimistic, favouring the baselines); 0.5 is the pessimistic
    sensitivity the sweep also reports.

    Returns ``(dict[int, np.ndarray], dict[int, int] offsets, list[str] which)``
    where ``which[c]`` names the mobility pattern client ``c`` drew.
    """
    traces, pats = load_archive(path)
    keep = [i for i, p in enumerate(pats)
            if (patterns is None or p in patterns) and len(traces[i]) >= min_samples]
    if not keep:
        raise ValueError("no traces match patterns=%r with >=%d samples"
                         % (patterns, min_samples))

    # A stream of its own (offset 20180614 = the dataset's publication year and
    # MMSys dates), so that changing the trace assignment never perturbs the
    # Markov chains, the client sampler, or the architecture draws.
    rng = np.random.default_rng([seed, 20180614])
    replace = len(keep) < num_clients
    pick = rng.choice(keep, size=num_clients, replace=replace)

    client_traces, offsets, which = {}, {}, []
    for c, i in enumerate(pick):
        t = traces[int(i)]
        client_traces[c] = t * float(ul_factor)
        offsets[c] = int(rng.integers(len(t)))
        which.append(pats[int(i)])
    return client_traces, offsets, which

# ---------------------------------------------------------------------------
# Sprout uplink traces (Winstein et al., NSDI 2013), via the Mahimahi archive.
#
# These are the only measured UPLINK capacity series this project could find in
# the public domain, and they exist because the Saturator drove both directions
# to saturation and logged when each packet crossed. Every other archive we
# examined -- the Irish 4G set, its 5G successor, the FL-over-5G testbed -- was
# collected while the phone DOWNLOADED, so its uplink column is acknowledgement
# traffic rather than capacity.
#
# They are replayed alongside the Irish archive, not instead of it: eight drive
# tests from 2013 have neither its variety of mobility nor its size, and the
# Irish set has neither their direction. A claim that survives both is not
# resting on either one's weakness.
# ---------------------------------------------------------------------------

SPROUT_ARCHIVE = os.path.join("analysis", "data", "sprout_uplink.npz")
_SPROUT_CANDIDATES = (os.path.join(_CODE_ROOT, SPROUT_ARCHIVE),
                      os.path.join(os.path.dirname(_CODE_ROOT), SPROUT_ARCHIVE))


def sprout_path(path=None):
    """Same two-depth probe as archive_path: the repository is checked out at
    one depth locally and another on the cluster."""
    if path:
        return os.path.abspath(path)
    for c in _SPROUT_CANDIDATES:
        if os.path.exists(c):
            return c
    return _SPROUT_CANDIDATES[-1]


def load_sprout_archive(path=None):
    """Return ``(list[np.ndarray] of bits/sec, list[str] names)``."""
    p = sprout_path(path)
    if not os.path.exists(p):
        raise FileNotFoundError(
            "no uplink archive at %s -- run "
            "`python3 analysis/fetch_mahimahi.py` to build it" % p)
    z = np.load(p, allow_pickle=False)
    kbps, offsets, names = z["kbps"], z["offsets"], z["names"]
    traces = [kbps[offsets[i]:offsets[i + 1]].astype(float) * 1e3
              for i in range(len(names))]
    return traces, [str(n) for n in names]


def load_sprout_client_traces(num_clients, seed=0, path=None, min_samples=120):
    """One measured uplink trace per client, with independent start offsets.

    There are eight traces and typically twenty clients, so assignment is WITH
    replacement and the start offsets do the separating: two clients on the same
    trace enter it at different points and so do not move in lockstep. That is
    weaker than the Irish archive's twenty distinct links and is the price of
    measuring the right direction.
    """
    traces, names = load_sprout_archive(path)
    keep = [i for i, t in enumerate(traces) if len(t) >= min_samples]
    if not keep:
        raise ValueError("no uplink trace reaches %d samples" % min_samples)

    rng = np.random.default_rng([seed, 20130401])
    pick = rng.choice(keep, size=num_clients,
                      replace=len(keep) < num_clients)
    client_traces, offsets, which = {}, {}, []
    for c, i in enumerate(pick):
        t = traces[int(i)]
        client_traces[c] = t
        offsets[c] = int(rng.integers(len(t)))
        which.append(names[int(i)])
    return client_traces, offsets, which
