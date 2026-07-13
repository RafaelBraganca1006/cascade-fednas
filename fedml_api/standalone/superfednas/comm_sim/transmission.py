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


def clip_arch_config(arch, tier):
    """Intersection α ∧ tier of a trained arch with a ladder tier.

    Middle clients under TS_all_random train *arbitrary* elastic archs (per-
    stage d, per-block e), not the configured tiers — so a client's shell
    ladder is the tier-clipped version of its own arch: d = min per stage,
    e = min per block. The result is a valid OFA config, and the clipped
    ladder is nested by construction:  α∧tier_0 ⊆ α∧tier_1 ⊆ ... ⊆ α.
    Keys other than d/e (e.g. w) are passed through from the arch.
    """

    def _min(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return min(a, b)

    def _combine(a_val, t_val):
        a_list = a_val if isinstance(a_val, (list, tuple)) else None
        t_list = t_val if isinstance(t_val, (list, tuple)) else None
        if a_list is not None and t_list is not None:
            assert len(a_list) == len(t_list), (a_val, t_val)
            return [_min(x, y) for x, y in zip(a_list, t_list)]
        if a_list is not None:
            return [_min(x, t_val) for x in a_list]
        if t_list is not None:
            return [_min(a_val, y) for y in t_list]
        return _min(a_val, t_val)

    clipped = dict(arch)
    for key in ("d", "e"):
        if key in arch and key in tier:
            clipped[key] = _combine(arch[key], tier[key])
    return clipped


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


class ShardPlan:
    """A carrier client's upload: whole supernet *blocks* at full width.

    Stage 3 sharding works at block granularity: a unit is one supernet block
    id, and carrying it means uploading that block's full-width parameters
    (convs + bns) from an XL-trained update. Units are independent — unlike
    shells there is no prefix semantics; each unit that fits the budget is
    individually usable, because aggregation is a per-coordinate average and
    the rest of the supernet keeps its coverage from the shell path.

    units: list of (unit_id, payload_bits) in send order (the allocator
    orders them by coverage debt, most-starved first).
    """

    def __init__(self, units):
        assert len(units) > 0
        self.units = list(units)

    def total_bits(self):
        return sum(bits for _, bits in self.units)


def truncate_units(plan, budget_bits, bandwidth_bps, window_s):
    """Cut a ShardPlan at the upload budget: whole units count, a half-sent
    unit is wire waste. Returns (DeliveryReport, delivered_unit_ids)."""
    remaining = max(float(budget_bits), 0.0)
    attempted = 0.0
    delivered_bits = 0.0
    delivered_ids = []
    for unit_id, bits in plan.units:
        if bits <= remaining:
            delivered_ids.append(unit_id)
            delivered_bits += bits
            attempted += bits
            remaining -= bits
        else:
            attempted += remaining
            remaining = 0.0
            break
    time_used = attempted / bandwidth_bps if bandwidth_bps > 0 else float(window_s)
    report = DeliveryReport(
        delivered_tier=0 if delivered_ids else -1,
        delivered_config=None,
        bits_attempted=attempted,
        bits_delivered=delivered_bits,
        time_used=min(time_used, float(window_s)),
        completed_plan=len(delivered_ids) == len(plan.units),
    )
    return report, delivered_ids


def build_clipped_plan(arch_config, tier_configs, payload_bits_fn, order=None):
    """Build one client's TransmissionPlan: its trained arch α clipped by the
    tier ladder (see clip_arch_config).

    tier_configs: ladder configs ascending by (unclipped) size — nestedness of
        the ladder is verified offline by verify_nesting.py, and it implies the
        clipped ladder is nested/ascending too.
    payload_bits_fn: config -> payload bits (the trainer passes a closure over
        server_model.get_subnet + CommSimulator.payload_bits, so the size used
        to plan matches the size charged at transmit time).

    The top of the clipped ladder is α itself (α ⊆ max tier), so completing
    the plan delivers exactly the trained update.
    """
    tiers = []
    for tier_cfg in tier_configs:
        clipped = clip_arch_config(arch_config, tier_cfg)
        tiers.append((payload_bits_fn(clipped), clipped))
    return TransmissionPlan(tiers, train_tier=len(tiers) - 1, order=order)


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
