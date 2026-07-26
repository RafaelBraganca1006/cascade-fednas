"""Per-client uplink bandwidth models for the dynFedNAS communication simulation.

Phase 1 of the thesis harness. A ``BandwidthModel`` answers one question:

    bandwidth(client_idx, round_num) -> available uplink in bits/sec.

The value is used by :class:`CommSimulator` together with the assigned subnet's
payload size and the round deadline to decide whether a client's update arrives
in time or is dropped. Models are pluggable so that a synthetic (Markov) source
can be swapped for real measured uplink traces (Irish 4G LTE dataset, Raca et
al., MMSys 2018) without touching the trainer.

All models are deterministic given their seed, and ``bandwidth(c, r)`` does not
depend on call order (it is precomputed per client), so runs are reproducible
regardless of which clients are sampled in which round.
"""

from abc import ABC, abstractmethod
import numpy as np


class BandwidthModel(ABC):
    @abstractmethod
    def bandwidth(self, client_idx, round_num):
        """Available uplink bandwidth (bits/sec) for a client in a given round."""
        raise NotImplementedError


class MarkovBandwidth(BandwidthModel):
    """Per-client discrete-state Markov chain over bandwidth levels.

    Each client has its own chain (seeded independently) that transitions once
    per round, producing a realistic, "sticky" fluctuating uplink: a bad link
    tends to stay bad for a while, then recovers. The whole trajectory is
    precomputed up front for reproducibility.

    Parameters
    ----------
    num_clients : int
    num_rounds : int          total communication rounds (trajectory length)
    states_bps : list[float]  bandwidth of each state in bits/sec
    transition : 2D array     row-stochastic SxS transition matrix
    seed : int
    jitter_sigma : float      log-normal spread WITHIN a state (0 = off, the
                              legacy discrete behaviour). See ``tiered_continuous``.
    """

    def __init__(self, num_clients, num_rounds, states_bps, transition, seed=0,
                 per_client_transition=None, client_tiers=None, jitter_sigma=0.0):
        self.states = np.asarray(states_bps, dtype=float)
        num_states = len(self.states)

        def _check(P):
            P = np.asarray(P, dtype=float)
            assert P.shape == (num_states, num_states), "transition must be SxS"
            assert np.allclose(P.sum(axis=1), 1.0), "transition rows must sum to 1"
            return P

        if per_client_transition is not None:
            assert len(per_client_transition) == num_clients
            matrices = [_check(P) for P in per_client_transition]
        else:
            matrices = [_check(transition)] * num_clients

        self.client_tiers = client_tiers  # None, or per-client profile index
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.jitter_sigma = float(jitter_sigma)
        self._traj = np.zeros((num_clients, num_rounds), dtype=np.int64)
        for c in range(num_clients):
            P = matrices[c]
            rng = np.random.default_rng([seed, c])  # independent per client
            s = int(rng.integers(num_states))       # random initial state
            for r in range(num_rounds):
                self._traj[c, r] = s
                s = int(rng.choice(num_states, p=P[s]))

        # Realized bandwidth per (client, round), precomputed so bandwidth() is
        # order-independent (the simulator queries the same (c, r) more than once
        # per round -- once when the sampler gets its capacity picture, again at
        # transmit time -- and those MUST agree, or a client would be sized
        # against one bandwidth and charged at another).
        self._bw = self.states[self._traj].astype(float)
        if self.jitter_sigma > 0:
            # Within-state variation. Drawn from a SEPARATE rng stream per client
            # (offset 104729) so the chain draws above are untouched: with
            # jitter_sigma=0 not a single extra random number is consumed, and
            # default()/tiered() stay bit-identical to their pre-jitter
            # trajectories (asserted in selftest_tiered_bandwidth.py).
            # Median-normalized (lognormal's mean is exp(sigma^2/2), not 1) so
            # the jittered profile has the SAME mean bandwidth as its discrete
            # counterpart and only differs in variance -- otherwise a comparison
            # between profiles would confound spread with a bandwidth increase.
            # Clipped at +-2 sigma in log space to keep tails physical.
            norm = np.exp(0.5 * self.jitter_sigma ** 2)
            lo, hi = np.exp(-2.0 * self.jitter_sigma), np.exp(2.0 * self.jitter_sigma)
            for c in range(num_clients):
                jrng = np.random.default_rng([seed, c, 104729])
                mult = jrng.lognormal(0.0, self.jitter_sigma, num_rounds) / norm
                self._bw[c] *= np.clip(mult, lo, hi)

    def bandwidth(self, client_idx, round_num):
        r = min(round_num, self.num_rounds - 1)  # clamp if run overshoots
        return float(self._bw[client_idx, r])

    def state(self, client_idx, round_num):
        r = min(round_num, self.num_rounds - 1)
        return int(self._traj[client_idx, r])

    @classmethod
    def default(cls, num_clients, num_rounds, seed=0):
        """A sensible 3-state mobile profile: good WiFi / congested 4G / near-outage.

        Sticky states (a link that is bad tends to stay bad), so drops are
        temporally correlated per client — the realistic case the thesis targets.
        """
        mbps = 1e6
        states_bps = [50 * mbps, 10 * mbps, 2 * mbps]  # good, medium, bad
        transition = [
            [0.80, 0.15, 0.05],  # from good
            [0.30, 0.50, 0.20],  # from medium
            [0.10, 0.40, 0.50],  # from bad (sticky)
        ]
        return cls(num_clients, num_rounds, states_bps, transition, seed=seed)

    # Connectivity profiles for the heterogeneous ("tiered") setting: same
    # bandwidth states, different dwell dynamics. Tier 0 = persistently poor
    # (rural / congested — mostly 2 Mbps), 1 = average (the .default matrix),
    # 2 = good (mostly 50 Mbps). Persistent per-client quality is what makes
    # the inclusion-gap and bandwidth-data correlation experiments possible;
    # .default has identical statistics for every client so no client is
    # persistently disadvantaged.
    TIER_NAMES = ("poor", "average", "good")
    TIER_TRANSITIONS = (
        [  # poor: sticky-bad
            [0.30, 0.40, 0.30],
            [0.10, 0.50, 0.40],
            [0.05, 0.25, 0.70],
        ],
        [  # average: the .default dynamics
            [0.80, 0.15, 0.05],
            [0.30, 0.50, 0.20],
            [0.10, 0.40, 0.50],
        ],
        [  # good: sticky-good
            [0.90, 0.08, 0.02],
            [0.50, 0.40, 0.10],
            [0.30, 0.40, 0.30],
        ],
    )

    @classmethod
    def tiered(cls, num_clients, num_rounds, client_tiers, seed=0):
        """Heterogeneous connectivity: ``client_tiers[c]`` picks each client's
        profile (0=poor, 1=average, 2=good). States match .default so payload
        feasibility thresholds are unchanged; only per-client dwell differs."""
        assert len(client_tiers) == num_clients
        mbps = 1e6
        states_bps = [50 * mbps, 10 * mbps, 2 * mbps]
        per_client = [cls.TIER_TRANSITIONS[t] for t in client_tiers]
        return cls(
            num_clients, num_rounds, states_bps, cls.TIER_TRANSITIONS[1],
            seed=seed, per_client_transition=per_client,
            client_tiers=list(client_tiers),
        )

    @classmethod
    def tiered_continuous(cls, num_clients, num_rounds, client_tiers, seed=0,
                          jitter_sigma=0.45):
        """``tiered`` with continuous within-state variation (E6b, 2026-07-26).

        Motivation (analysis/pick_deadline.py, DIRECTION §3.2b): with three
        discrete states there are only three possible budgets at any deadline T,
        so at most three of the {dead, S, M, L, XL} outcomes can ever occur --
        at EVERY T at least one tier is unreachable by every client in every
        round. That is the structural cause of the degenerate assignment
        problem the policy audit found, and no choice of T fixes it. Adding
        spread within a state makes the budget continuous, so the whole tier
        ladder becomes reachable and a size/cap decision has something to
        decide.

        This is a step TOWARD realism, not away from it: measured 4G uplink
        varies continuously within a connectivity regime -- the discrete states
        were an abstraction that suited a binary-drop mechanism study and is too
        coarse for a scheduling study. ``jitter_sigma`` is the one free
        parameter and should be calibrated against the within-regime variance of
        the Irish traces once E9 lands; until then 0.45 is an exploratory
        starting point (same honesty as lambda=0.1 for Lyapunov), chosen so the
        +-2 sigma band spans roughly a 6x range around each state's median.

        Keeps the sticky Markov chain untouched -- that persistence is the
        learnable structure the selection policy depends on (a client good at
        r-1 is good at r with p ~ 0.8-0.9); only the value within a state
        varies. Mean bandwidth per client is unchanged vs ``tiered`` (the
        multiplier is median-normalized), so profile comparisons isolate
        variance.
        """
        assert len(client_tiers) == num_clients
        mbps = 1e6
        states_bps = [50 * mbps, 10 * mbps, 2 * mbps]
        per_client = [cls.TIER_TRANSITIONS[t] for t in client_tiers]
        return cls(
            num_clients, num_rounds, states_bps, cls.TIER_TRANSITIONS[1],
            seed=seed, per_client_transition=per_client,
            client_tiers=list(client_tiers), jitter_sigma=jitter_sigma,
        )


def assign_connectivity_tiers(client_class_counts, fractions=(0.3, 0.4, 0.3),
                              corr=0.0, seed=0):
    """Assign a connectivity tier (0=poor, 1=average, 2=good) to each client,
    optionally correlated with WHAT DATA the client holds.

    corr = 0.0: tiers are a seeded random assignment (independent of data).
    corr = 1.0: clients are ranked by the center of mass of their label
        distribution — clients concentrating low-index classes become the
        poorly connected ones (digital-divide stress test, DIRECTION §5:
        methods without unbiased correction develop per-class accuracy holes).
    0 < corr < 1 interpolates (rank blend).

    client_class_counts: dict[client_idx] -> per-class sample counts (array).
    Returns list[int] of tiers, index = client_idx.
    """
    n = len(client_class_counts)
    counts = [np.asarray(client_class_counts[c], dtype=float) for c in range(n)]
    com = np.array([
        (cnt * np.arange(len(cnt))).sum() / max(cnt.sum(), 1.0) for cnt in counts
    ])
    rng = np.random.default_rng([seed, 7919])  # distinct stream from the chains
    rank_data = np.argsort(np.argsort(com))          # low class center -> low rank
    rank_rand = np.argsort(rng.permutation(n))
    score = corr * rank_data + (1.0 - corr) * rank_rand
    order = np.argsort(score)  # first = lowest score -> poor tier

    n_poor = int(round(fractions[0] * n))
    n_avg = int(round(fractions[1] * n))
    tiers = [None] * n
    for pos, c in enumerate(order):
        tiers[c] = 0 if pos < n_poor else (1 if pos < n_poor + n_avg else 2)
    return tiers


class TraceReplayBandwidth(BandwidthModel):
    """Replay real bandwidth traces (Irish 4G LTE dataset — measured uplink
    throughput across five mobility patterns; Raca et al., MMSys 2018).

    Placeholder for Phase 5. Each client is assigned a trace (a 1D array of
    throughput samples in bits/sec); ``bandwidth(c, r)`` indexes into it,
    looping/segmenting as needed to cover all rounds. Implemented once the
    synthetic mechanism is validated.
    """

    def __init__(self, client_traces_bps):
        # client_traces_bps: dict[int, np.ndarray] of per-client throughput series
        self.traces = {c: np.asarray(t, dtype=float) for c, t in client_traces_bps.items()}

    def bandwidth(self, client_idx, round_num):
        trace = self.traces[client_idx]
        return float(trace[round_num % len(trace)])
