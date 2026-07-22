"""Greedy assignment policy -- DIRECTION.md §3.2, Lemma 1.

Under oracle bandwidth + IID data + lambda=0 (cross-context poisoning
negligible), assigning every client the global max tier is provably optimal:
IID means no client's data is worth more than another's, so only available
budget differentiates clients, and nested transmission's truncation absorbs
any overshoot for free (no total-loss penalty, unlike all_or_nothing). The
exchange argument: swapping any assignment for "give them the max tier
instead" can never decrease the debt-weighted bits delivered, since
truncation degrades gracefully to whatever the client's budget actually
supports.

This makes greedy's implementation a genuine constant function -- it does
not need bandwidth, debt, or data-utility context to be optimal in this
regime, which is exactly what the self-test in selftest_policy.py checks.
It stops being optimal once lambda > 0 (E3-dependent); that regime is
lyapunov.py / bandit.py's job.
"""

from .base import AssignmentContext, AssignmentPolicy, DeliveryFeedback, TierOption


class AssignMaxPolicy(AssignmentPolicy):
    """Always assigns the largest available tier. See module docstring:
    this is the lambda=0 optimum (Lemma 1), not a naive placeholder."""

    def assign(self, context: AssignmentContext) -> TierOption:
        if not context.tier_options:
            raise ValueError("AssignmentContext.tier_options is empty")
        # tier_options is sorted ascending by payload (base_server_model's
        # _cap_tier_bits convention) -- the last entry is the max tier.
        return max(context.tier_options, key=lambda t: t.payload_bits)

    def update(self, feedback: DeliveryFeedback) -> None:
        pass  # stateless: no debt table, no learned weights
