"""Shared interface for subnet-assignment policies (DIRECTION.md §3.2, retargeted
2026-07-22: the policy decides "who trains what tier" under nested transmission,
not shard-carrier selection).

Three implementations share this interface and live side by side permanently
(TODO.md "Three-policy architecture") -- greedy.py, lyapunov.py, bandit.py.
Selected by --assignment_policy; never edited in place once landed, so a
single flag flip reproduces any of the three under identical seeds/config.

The objective all three optimize for (once lambda is set by E3, DIRECTION
§3.2): maximize debt-weighted NATIVE-context delivered bits minus lambda
times cross-context (truncated) delivered bits. lambda=0 (if E3 finds
poisoning negligible) makes "always assign the max tier" trivially optimal
-- that is exactly what greedy.py implements, with no dependency on E3's
result. lyapunov.py and bandit.py need lambda > 0 to be meaningful and are
stubs until then.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TierOption:
    """One entry of the client's tier menu (mirrors base_server_model's
    _cap_tier_bits: sorted ascending by payload size)."""

    tier_id: str  # key into diverse_subnets, e.g. "0".."3" (S..XL)
    config: dict  # the {"d":..., "e":...} architecture config
    payload_bits: float


@dataclass
class AssignmentContext:
    """Everything an AssignmentPolicy may use to decide a client's tier this
    round. Fields a policy doesn't need are simply ignored (greedy ignores
    everything but tier_options; lyapunov/bandit will use bw/debt/utility).

    round_num, client_idx: identify the decision point.
    tier_options: the client's tier menu, ascending by payload size.
    bw_estimate_bps: this client's (possibly lagged) bandwidth estimate,
        bits/sec, or None if unavailable.
    upload_window_s: seconds available for upload this round.
    debt_by_tier: {tier_id: staleness/debt Q(t)} for the shells this tier
        would touch -- aggregated over ALL clients (Lyapunov's queue: "how
        long since ANYONE refreshed this tier"). None until the assignment-
        level debt table exists.
    client_tier_debt: {tier_id: staleness of THIS client at this tier --
        "how long since THIS client specifically refreshed this tier"}.
        Added 2026-07-26 after the non-IID pilot (results/README.md) found
        blind beats both Lyapunov and the bandit there: debt_by_tier is
        data-blind by construction (tracks WHEN a tier was served, never
        WHICH client), and under non-IID that hides which classes a tier's
        recent contributions actually represent -- blind's uniform random
        draw incidentally rotates client-tier exposure, engineered debt
        doesn't. This field lets a policy target that directly instead of
        by accident. Only the bandit consumes it (see bandit.py); Lyapunov
        is deliberately left untouched -- it's the honest non-learned
        baseline, not a target for iteration. None if not wired.
    data_utility: the client's last self-reported (config, loss), or None.
    """

    round_num: int
    client_idx: int
    tier_options: List[TierOption]
    bw_estimate_bps: Optional[float] = None
    upload_window_s: Optional[float] = None
    debt_by_tier: Optional[Dict[str, float]] = None
    client_tier_debt: Optional[Dict[str, float]] = None
    data_utility: Optional[dict] = None


@dataclass
class DeliveryFeedback:
    """Post-round outcome, fed back via update() so stateful policies
    (lyapunov's queues, the bandit's weights) can learn. Stateless policies
    (greedy) implement update() as a no-op."""

    round_num: int
    client_idx: int
    assigned_tier: str
    delivered_tier: Optional[str]  # None if nothing delivered
    native_context: bool  # True iff delivered_tier == assigned_tier (no truncation)
    native_bits: float
    cross_context_bits: float


class AssignmentPolicy(ABC):
    @abstractmethod
    def assign(self, context: AssignmentContext) -> TierOption:
        """Ex-ante decision: which tier does this client train this round."""
        raise NotImplementedError

    def update(self, feedback: DeliveryFeedback) -> None:
        """Post-round feedback hook. Default no-op (stateless policies like
        greedy don't need it); stateful policies override this."""
        pass
