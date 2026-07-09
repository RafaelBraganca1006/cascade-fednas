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
    """

    def __init__(self, num_clients, num_rounds, states_bps, transition, seed=0):
        self.states = np.asarray(states_bps, dtype=float)
        P = np.asarray(transition, dtype=float)
        num_states = len(self.states)
        assert P.shape == (num_states, num_states), "transition must be SxS"
        assert np.allclose(P.sum(axis=1), 1.0), "transition rows must sum to 1"

        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self._traj = np.zeros((num_clients, num_rounds), dtype=np.int64)
        for c in range(num_clients):
            rng = np.random.default_rng([seed, c])  # independent per client
            s = int(rng.integers(num_states))       # random initial state
            for r in range(num_rounds):
                self._traj[c, r] = s
                s = int(rng.choice(num_states, p=P[s]))

    def bandwidth(self, client_idx, round_num):
        r = min(round_num, self.num_rounds - 1)  # clamp if run overshoots
        return float(self.states[self._traj[client_idx, r]])

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
