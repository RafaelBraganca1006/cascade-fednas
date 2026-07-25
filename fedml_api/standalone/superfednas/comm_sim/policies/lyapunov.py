"""Lyapunov drift-plus-penalty assignment policy -- DIRECTION.md §3.2.

Implemented 2026-07-24, after E3 (results/README.md) fixed the SHAPE of
lambda: cross-context delivery is net beneficial almost everywhere, with a
real but small cost concentrated at the S shell specifically (removing
cross-context helped S by +0.94 but collapsed M/L/XL, spread 8.7->1.8) --
not a broad "cross-context is bad" penalty. The score below encodes exactly
that: assigning tier i earns the full debt-weighted value of every shell
0..i; the ONLY discount is a flat lambda penalty on shell 0's (S's) credit,
applied whenever the assignment reaches beyond S (i.e. whenever S could
plausibly arrive via truncation from a bigger trained architecture, per
E3's finding). M/L/XL shells never carry a penalty.

Deliberately does NOT model a bandwidth "ceiling" / truncation-probability
estimate. nested transmission already delivers whatever fits regardless of
the assigned tier -- the deadline truncates ex-post (DIRECTION §3.1's
prediction-free-delivery property), so assigning bigger never produces LESS
delivered content than assigning smaller, only changes whether the result
is native or cross-context. The policy's job is purely to balance shell
debt against the S-penalty, not to predict what will arrive.
bw_estimate_bps/upload_window_s are available on the context (other
policies may use them) but unused here by design; a bandwidth-aware
refinement is future work, not required for this to be a correct v1.

Degenerates EXACTLY to greedy's AssignMaxPolicy (Lemma 1) when lambda=0:
every additional shell's debt-weighted value is >= 0, so the score is
non-decreasing in tier index, and ties are broken toward the larger tier --
always the max tier. Executable as a self-test (selftest_policy.py).
"""

from .base import AssignmentContext, AssignmentPolicy, DeliveryFeedback, TierOption


class LyapunovPolicy(AssignmentPolicy):
    def __init__(self, lam: float = None):
        if lam is None:
            raise NotImplementedError(
                "LyapunovPolicy needs lambda from the E3 gradient-context "
                "ablation (DIRECTION §3.2) -- see module docstring."
            )
        self.lam = lam

    def assign(self, context: AssignmentContext) -> TierOption:
        if not context.tier_options:
            raise ValueError("AssignmentContext.tier_options is empty")
        # Defensively sort ascending by payload rather than trusting the
        # caller's order (greedy is order-independent via max(); shell-size
        # decomposition below needs ascending order to be correct, so make
        # that a guarantee of THIS function, not an assumption on the input).
        tiers = sorted(context.tier_options, key=lambda t: t.payload_bits)
        if context.debt_by_tier is None:
            raise ValueError(
                "LyapunovPolicy requires AssignmentContext.debt_by_tier -- "
                "wire the tier-debt table through policy_assign_sample "
                "(superfednas_trainer.py's _tier_last_round / "
                "_mark_tier_debt_served)."
            )

        debts = [context.debt_by_tier.get(t.tier_id, 0.0) for t in tiers]
        shell_sizes = self._shell_sizes(tiers)

        best_idx, best_value = 0, None
        for i in range(len(tiers)):
            reaches_beyond_s = i > 0
            value = debts[0] * shell_sizes[0] * (
                (1.0 - self.lam) if reaches_beyond_s else 1.0
            )
            for k in range(1, i + 1):
                value += debts[k] * shell_sizes[k]
            # tie-break toward the LARGER tier -- matches greedy's Lemma-1
            # behavior exactly when lambda=0 (all ties resolve to max tier).
            if best_value is None or value >= best_value:
                best_value, best_idx = value, i
        return tiers[best_idx]

    @staticmethod
    def _shell_sizes(tiers):
        """tiers is ascending by cumulative payload; shell k's own size is
        the marginal payload over shell k-1 (shell 0 = tier 0's full payload)."""
        sizes = [tiers[0].payload_bits]
        for i in range(1, len(tiers)):
            sizes.append(tiers[i].payload_bits - tiers[i - 1].payload_bits)
        return sizes

    def update(self, feedback: DeliveryFeedback) -> None:
        pass  # stateless: debt lives in the trainer's _tier_last_round table
