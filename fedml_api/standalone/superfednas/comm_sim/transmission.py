"""Transmission plans and delivery reports for dynFedNAS (v2 mechanisms).

Replaces the binary drop decision with an explicit model of *what goes over the
wire and in which order*, so a deadline can truncate an upload instead of
voiding it. Three transmission modes:

  - ``all_or_nothing``  legacy Phase 1-3 behaviour: the whole assigned subnet
                        either fits within the deadline or the update is lost.
  - ``nested``          the upload is ordered as nested shells (S, M\\S, L\\M,
                        XL\\L, ...); the deadline cuts the stream and every
                        *complete* prefix of shells is a valid update of a
                        smaller tier (anytime property).
  - ``nested_shard``    nested, plus the top shell split across clients
                        (Stage 3; ShardSpec is a placeholder until then).

This module is pure bookkeeping over bit counts and tier configs — no torch —
so it is unit-testable next to ``selftest_capacity_aware.py``.
"""

TX_MODES = ("all_or_nothing", "nested", "nested_shard")


class TransmissionPlan:
    """What the server instructs one client to send this round.

    tiers: list of (payload_bits, config) for the *nested* tier ladder,
        ascending by size — same structure as the capacity_aware sampler's
        ``_cap_tier_bits``. The client trains ``tiers[train_tier]`` and
        uploads its update as shells derived from the ladder.
    train_tier: index into ``tiers`` of the subnet the client actually trains
        (the upload can only ever contain shells up to this tier).
    order: shell indices in send order. Defaults to bottom-up (0, 1, ..),
        which preserves the anytime property exactly; other orders are a
        scheduler decision (a tier is only complete once ALL its shells
        arrived).
    shard: reserved for Stage 3 (top-shell sharding); must be None for now.
    """

    def __init__(self, tiers, train_tier, order=None, shard=None):
        assert len(tiers) > 0
        assert 0 <= train_tier < len(tiers)
        bits = [t[0] for t in tiers]
        assert bits == sorted(bits), "tier ladder must be ascending"
        self.tiers = tiers
        self.train_tier = train_tier
        self.order = list(order) if order is not None else list(range(train_tier + 1))
        assert sorted(self.order) == sorted(set(self.order)), "duplicate shells"
        assert all(0 <= s <= train_tier for s in self.order)
        assert shard is None, "sharding lands in Stage 3"
        self.shard = shard

    def shell_bits(self, shell_idx):
        """Size of shell k = tier_k payload minus tier_{k-1} payload."""
        if shell_idx == 0:
            return self.tiers[0][0]
        return self.tiers[shell_idx][0] - self.tiers[shell_idx - 1][0]

    def total_bits(self):
        """Bits of the full planned upload (all shells up to train_tier)."""
        return self.tiers[self.train_tier][0]


class DeliveryReport:
    """What the deadline let through for one client.

    delivered_tier: index of the largest tier whose shells ALL arrived,
        or -1 if not even the smallest tier completed.
    delivered_config: that tier's config (None when delivered_tier == -1).
    bits_attempted: bits pushed onto the wire (capped by budget).
    bits_delivered: bits belonging to complete tiers (usable by aggregation).
    bits_wasted: attempted - delivered (truncated shell remainders; in
        all_or_nothing mode, everything transmitted before a timeout).
    time_used: seconds of the upload window consumed.
    completed_plan: the full planned upload arrived (no truncation).
    """

    def __init__(
        self,
        delivered_tier,
        delivered_config,
        bits_attempted,
        bits_delivered,
        time_used,
        completed_plan,
    ):
        self.delivered_tier = delivered_tier
        self.delivered_config = delivered_config
        self.bits_attempted = float(bits_attempted)
        self.bits_delivered = float(bits_delivered)
        self.bits_wasted = float(bits_attempted) - float(bits_delivered)
        self.time_used = float(time_used)
        self.completed_plan = completed_plan

    def delivers_nothing(self):
        """True when no complete tier arrived — the client need not train
        (the reduced, still-valid form of the pre-training skip)."""
        return self.delivered_tier < 0


def truncate(plan, budget_bits, bandwidth_bps, window_s):
    """Cut the planned shell stream at the upload budget (nested modes).

    Walks shells in ``plan.order``; a shell counts only if it fits completely
    in the remaining budget (a half-arrived shell is wire waste — Stage 3
    turns that remainder into shard opportunity). The delivered tier is the
    largest k such that shells 0..k all arrived.
    """
    remaining = max(float(budget_bits), 0.0)
    attempted = 0.0
    arrived = set()
    for shell in plan.order:
        bits = plan.shell_bits(shell)
        if bits <= remaining:
            arrived.add(shell)
            attempted += bits
            remaining -= bits
        else:
            # stream cut mid-shell: those bits crossed the wire but are unusable
            attempted += remaining
            remaining = 0.0
            break

    delivered_tier = -1
    while delivered_tier + 1 in arrived:
        delivered_tier += 1
    delivered_bits = plan.tiers[delivered_tier][0] if delivered_tier >= 0 else 0.0
    delivered_config = plan.tiers[delivered_tier][1] if delivered_tier >= 0 else None
    completed = arrived == set(plan.order)
    time_used = (
        attempted / bandwidth_bps if bandwidth_bps > 0 else float(window_s)
    )
    return DeliveryReport(
        delivered_tier=delivered_tier,
        delivered_config=delivered_config,
        bits_attempted=attempted,
        bits_delivered=delivered_bits,
        time_used=min(time_used, float(window_s)),
        completed_plan=completed,
    )


def all_or_nothing(payload_bits, budget_bits, bandwidth_bps, window_s):
    """Legacy semantics as a DeliveryReport: the whole payload or nothing.

    On a timeout the client still transmitted until the window closed, so the
    wire waste is min(payload, budget) — the bits that crossed and got voided.
    """
    payload_bits = float(payload_bits)
    success = payload_bits <= float(budget_bits)
    if success:
        return DeliveryReport(
            delivered_tier=0,
            delivered_config=None,  # caller knows the assigned config
            bits_attempted=payload_bits,
            bits_delivered=payload_bits,
            time_used=payload_bits / bandwidth_bps if bandwidth_bps > 0 else 0.0,
            completed_plan=True,
        )
    return DeliveryReport(
        delivered_tier=-1,
        delivered_config=None,
        bits_attempted=min(payload_bits, max(float(budget_bits), 0.0)),
        bits_delivered=0.0,
        time_used=float(window_s),
        completed_plan=False,
    )
