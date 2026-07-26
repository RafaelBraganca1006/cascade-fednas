"""Client-SELECTION policies -- DIRECTION.md §3.2b, EXPERIMENTS.md E6b Stage 1.

Different decision surface from ``base.AssignmentPolicy``: that one answers
"which tier does this already-selected client train", this one answers "which
clients are in the round at all". The audit in DIRECTION §3.2b moved the policy
chapter here, because under nested transmission the assignment decision stopped
gating what gets delivered (truncation already delivers the largest affordable
prefix regardless of what was assigned) while selection still gates it
completely: a client whose budget is below the smallest tier delivers NOTHING,
and with the sticky "poor" profile those clients are dead for long, predictable
stretches. ``analysis/pick_deadline.py`` measures the size of that lever --
~35% of participation slots at the E6b canonical config.

The two probes here are deliberately NOT learned and deliberately extreme: each
pushes ONE signal as far as it goes, so Stage 1 can answer "does the who-
participates axis carry any signal?" before an RL is built on top of it.

**They are not ceilings.** An earlier draft called them that; a true upper bound
would need an oracle over each client's eventual contribution to accuracy, which
is not cheaply computable. ``OracleBandwidthSelector`` can land BELOW blind --
under non-IID it probably will, since "always pick the fastest" is precisely the
digital-divide pathology this thesis argues against. That outcome is a POSITIVE
result for the chapter: it proves the axis has large impact and that the naive
direction is destructive. What kills the retarget is the two probes TYING with
blind, not probe A losing.

Both probes read the true current-round bandwidth on purpose (they are
diagnostics, not deployable policies). The learned policy that follows them must
use ``--bw_estimate lagged`` and exploit the chain's stickiness instead.
"""

from abc import ABC, abstractmethod


class SelectionPolicy(ABC):
    """Chooses the participating clients for one round.

    Implementations must return exactly ``num_to_select`` distinct entries drawn
    from ``candidates``, and must be deterministic given the same inputs (the
    trainer reproduces rounds across arms).
    """

    @abstractmethod
    def select(self, round_num, candidates, num_to_select, ctx):
        """ctx: SelectionContext. Returns a list of client indices."""
        raise NotImplementedError

    def update(self, round_num, client_idx, delivered_bits, delivered_tier):
        """Post-round outcome hook. No-op for the stateless probes; the learned
        selector overrides it."""
        pass


class SelectionContext:
    """What a selection policy may look at before the round runs.

    bw_true_bps: {client_idx: this round's TRUE bandwidth} -- oracle, probes only.
    bw_est_bps:  {client_idx: the lagged estimate} -- what a deployable policy sees.
    min_tier_bits: payload of the smallest tier; a client whose budget is below
        this delivers nothing whatever it is assigned (the "dead slot" test).
    upload_window_s: seconds available for upload.
    delivered_counts: {client_idx: {tier_id: times this client has delivered at
        least this tier}} -- the coverage table the fairness probe maximises over.
    last_round_seen: {client_idx: last round this client delivered anything}.
    """

    def __init__(self, bw_true_bps=None, bw_est_bps=None, min_tier_bits=None,
                 upload_window_s=None, delivered_counts=None,
                 last_round_seen=None):
        self.bw_true_bps = bw_true_bps or {}
        self.bw_est_bps = bw_est_bps or {}
        self.min_tier_bits = min_tier_bits
        self.upload_window_s = upload_window_s
        self.delivered_counts = delivered_counts or {}
        self.last_round_seen = last_round_seen or {}

    def budget_bits(self, client_idx, oracle=True):
        bw = (self.bw_true_bps if oracle else self.bw_est_bps).get(client_idx)
        if bw is None or self.upload_window_s is None:
            return None
        return bw * self.upload_window_s

    def is_dead(self, client_idx, oracle=True):
        """True iff this client cannot complete even the smallest tier."""
        budget = self.budget_bits(client_idx, oracle=oracle)
        if budget is None or self.min_tier_bits is None:
            return False
        return budget < self.min_tier_bits


class OracleBandwidthSelector(SelectionPolicy):
    """Probe A -- the bandwidth signal at its extreme: top-K by true current
    bandwidth, maximising delivered bytes per round.

    Expected to WIN on delivered GB and mean accuracy under IID, and to LOSE on
    the inclusion gap (and plausibly on non-IID mean accuracy) by starving the
    persistently-poor clients entirely. Both directions are informative; see the
    module docstring.

    Ties broken by client index so the draw is reproducible across arms.
    """

    def select(self, round_num, candidates, num_to_select, ctx):
        ranked = sorted(
            candidates,
            key=lambda c: (-ctx.bw_true_bps.get(c, 0.0), c),
        )
        return ranked[:num_to_select]


class CoverageSelector(SelectionPolicy):
    """Probe B -- the staleness/coverage signal at its extreme: prefer the
    clients whose delivered coverage is lowest, i.e. maximise the MINIMUM
    delivered-update count across clients (§3.4's architecture-fairness
    objective, max-min rather than sum).

    Deliberately paired with a feasibility screen rather than ignoring bandwidth
    entirely: selecting a client that provably cannot deliver even the smallest
    tier this round improves nobody's coverage and just burns the slot, which is
    the exact waste the chapter is about. So starved-but-feasible clients come
    first; if fewer than num_to_select are feasible, the remaining slots fall
    back to the starved-but-infeasible ones (never returns short).

    Note the honest asymmetry vs Probe A: this one also consults oracle
    bandwidth, for the feasibility screen only, not for ranking.
    """

    def _total_delivered(self, ctx, client_idx):
        return sum(ctx.delivered_counts.get(client_idx, {}).values())

    def select(self, round_num, candidates, num_to_select, ctx):
        def starvation_key(c):
            # least-delivered first; among equals, the one unseen longest;
            # client index last so ties are deterministic
            return (
                self._total_delivered(ctx, c),
                ctx.last_round_seen.get(c, -1),
                c,
            )

        feasible = [c for c in candidates if not ctx.is_dead(c)]
        infeasible = [c for c in candidates if ctx.is_dead(c)]
        chosen = sorted(feasible, key=starvation_key)[:num_to_select]
        if len(chosen) < num_to_select:
            chosen += sorted(infeasible, key=starvation_key)[
                : num_to_select - len(chosen)
            ]
        return chosen


SELECTION_POLICIES = {
    "oracle_bandwidth": OracleBandwidthSelector,
    "coverage": CoverageSelector,
}
