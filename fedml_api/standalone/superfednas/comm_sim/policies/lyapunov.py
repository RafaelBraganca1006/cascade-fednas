"""Lyapunov drift-plus-penalty assignment policy -- DIRECTION.md §3.2.

STUB (2026-07-22): structurally complete (implements AssignmentPolicy so it
slots into the same --assignment_policy dispatch as greedy/bandit), but
assign()/update() raise NotImplementedError until:
  1. E3's ablation result sets lambda (the cross-context penalty rate) --
     without it there is no objective to optimize (see base.py docstring).
  2. The assignment-level debt table exists. Today only the shard-level
     `_unit_last_round` (per supernet BLOCK, used by _allocate_shards for
     carrier/unit selection) exists in the trainer. Lyapunov's virtual
     queues Q_u(t) are that same table, reinterpreted/aggregated to TIER
     granularity for this decision -- needs a small trainer-side change to
     expose it here (or a parallel tier-level table), not a new design.

Mechanics once unblocked (DIRECTION §3.2, item 2): each round, score every
candidate tier for a client as (debt-weighted native bits it would add) -
V * lambda * (expected cross-context bits if truncated below it), pick the
top-scoring tier. V is the standard Lyapunov queue-vs-penalty trade-off
knob. Degenerates to AssignMaxPolicy's behavior when lambda=0.
"""

from .base import AssignmentContext, AssignmentPolicy, DeliveryFeedback, TierOption


class LyapunovPolicy(AssignmentPolicy):
    def __init__(self, lam: float = None, v: float = 1.0):
        if lam is None:
            raise NotImplementedError(
                "LyapunovPolicy needs lambda from the E3 gradient-context "
                "ablation (DIRECTION §3.2) -- not yet available. See module "
                "docstring."
            )
        self.lam = lam
        self.v = v
        self._debt = {}  # tier_id -> Q_u(t), populated via update()

    def assign(self, context: AssignmentContext) -> TierOption:
        raise NotImplementedError(
            "LyapunovPolicy.assign: needs the tier-level debt table wired "
            "from the trainer (see module docstring, point 2)."
        )

    def update(self, feedback: DeliveryFeedback) -> None:
        raise NotImplementedError(
            "LyapunovPolicy.update: queue dynamics not yet implemented."
        )
