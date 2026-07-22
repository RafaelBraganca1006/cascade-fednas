"""Contextual bandit assignment policy -- DIRECTION.md §3.2, the policy-chapter
contribution (E6).

STUB (2026-07-22): structurally complete, but assign()/update() raise
NotImplementedError until LyapunovPolicy is working (this reuses its context
vector -- bandwidth history, coverage-debt summary, data-utility report --
and swaps the fixed drift-plus-penalty scoring rule for a learned linear
scorer over the same features) and E3 has set lambda for the reward.

Reward (mirrors Lyapunov's objective exactly, DIRECTION §3.2 item 3): dense,
observable every round -- debt-weighted native-context bits delivered minus
lambda times cross-context bits delivered. No sparse end-of-run accuracy
reward (a full run is 4-6h; cross-run policy gradient on a sparse reward is
infeasible -- this is why the reward must be dense and per-round).

Ladder (stop at whichever level lands, DIRECTION §3.2): (i) learned
bandwidth predictor feeding Lyapunov: (ii) THIS -- contextual bandit
directly over tier choice; (iii) joint long-horizon RL, only if (ii) lands
early. This stub targets (ii).
"""

from .base import AssignmentContext, AssignmentPolicy, DeliveryFeedback, TierOption


class ContextualBanditPolicy(AssignmentPolicy):
    def __init__(self, lam: float = None):
        if lam is None:
            raise NotImplementedError(
                "ContextualBanditPolicy needs lambda from the E3 gradient-"
                "context ablation (DIRECTION §3.2), same reward as "
                "LyapunovPolicy -- not yet available. See module docstring."
            )
        self.lam = lam
        self._weights = None  # linear scorer over context features, TBD

    def assign(self, context: AssignmentContext) -> TierOption:
        raise NotImplementedError(
            "ContextualBanditPolicy.assign: build on LyapunovPolicy's "
            "context wiring first (see module docstring)."
        )

    def update(self, feedback: DeliveryFeedback) -> None:
        raise NotImplementedError(
            "ContextualBanditPolicy.update: weight update rule not yet "
            "implemented."
        )
