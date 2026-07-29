"""Shell sharding (Stage 3): collective upload of the starving supernet region.

No single client can upload the XL update under a tight deadline (harsh: the
top shell alone, 149 MB, exceeds even a 50 Mbps client's 20 s budget). But the
server only needs the per-coordinate *average* of updates, and `add_subnet`
already accumulates whatever coordinates each client delivers. So designated
carrier clients train the XL subnet and upload only assigned *units* — whole
supernet blocks at full width. Collectively, the units cover the region that
individually fits nobody.

Granularity choice (IMPLEMENTATION_PLAN Stage 3.1): whole blocks, not channel
ranges. Uploading block b at XL width also refreshes b's narrower channels —
harmless extra coverage, and it makes the aggregation mask a plain key filter
(no slice arithmetic, no BN mismatch risk).

Pure-python allocator + wrapper here; the trainer owns carrier selection and
the debt table.
"""


def parse_block_id(key):
    """Supernet-side param key -> block id, or None for stem/classifier."""
    parts = key.split(".")
    if parts[0] == "blocks" and len(parts) > 1:
        return int(parts[1])
    return None


def block_payload_bits(param_numels, bytes_per_param=4):
    """{block_id: payload bits} from a {supernet key: numel} mapping."""
    bits = {}
    for key, numel in param_numels.items():
        block = parse_block_id(key)
        if block is not None:
            bits[block] = bits.get(block, 0) + numel * bytes_per_param * 8
    return bits


def allocate_units(unit_bits, carrier_budgets, debt_order):
    """Assign shard units to carriers, most-starved units first.

    unit_bits: {unit_id: bits}; carrier_budgets: {carrier: est. budget bits}
    (lagged bandwidth x upload window — an estimate, so allocation mistakes
    cost coverage this round, never correctness); debt_order: unit ids, most
    starved first. Greedy: each unit goes to the carrier with the most
    remaining budget that still fits it. Returns {carrier: [unit ids]}.
    """
    remaining = dict(carrier_budgets)
    assignment = {c: [] for c in carrier_budgets}
    for unit in debt_order:
        bits = unit_bits[unit]
        best = None
        for c, room in remaining.items():
            if bits <= room and (best is None or room > remaining[best]):
                best = c
        if best is not None:
            assignment[best].append(unit)
            remaining[best] -= bits
    return assignment


def client_block_bits(client_model, bytes_per_param=4):
    """{supernet_block_id: payload bits} of ONE client's (sub)model blocks.

    For the sub-floor shard fallback: a client whose budget fits no complete
    tier can still upload whole blocks of its own trained subnet. Sizes come
    from the client's parameter shapes (smaller than the full-width units in
    block_payload_bits), keyed by supernet block id via model_index.
    """
    bits = {}
    for key, tensor in client_model.state_dict().items():
        parts = key.split(".")
        if parts[0] != "blocks":
            continue
        supernet_block = client_model.model_index["blocks"][int(parts[1])]
        bits[supernet_block] = (
            bits.get(supernet_block, 0) + tensor.numel() * bytes_per_param * 8
        )
    return bits


def per_block_bits_of_config(client_model, bytes_per_param=4):
    """{supernet_block_id (or OTHER_BLOCK): payload bits} for ONE subnet.

    Like client_block_bits, but ALSO accounts for stem/classifier parameters
    under the OTHER_BLOCK sentinel, so the per-block totals sum exactly to the
    subnet's payload. The block stream needs that exactness -- if the units
    don't sum to the payload, the byte accounting diverges from the shell path
    and the two transmission modes stop being comparable.
    """
    from .transmission import OTHER_BLOCK

    bits = {}
    for key, tensor in client_model.state_dict().items():
        parts = key.split(".")
        if parts[0] == "blocks":
            unit = client_model.model_index["blocks"][int(parts[1])]
        else:
            unit = OTHER_BLOCK
        bits[unit] = bits.get(unit, 0) + tensor.numel() * bytes_per_param * 8
    return bits


def build_block_stream(tier_models, tier_configs, bytes_per_param=4,
                       block_order=None, rng=None):
    """Build the (shell, block) unit stream for one client (DIRECTION §3.1c).

    tier_models: the client's arch clipped by each ladder tier, ascending --
        i.e. exactly what build_clipped_plan produces, materialised as models.
    block_order: optional {shell_idx: [block ids in send order]}. Within a
        shell the blocks are INDEPENDENT (no prefix semantics), so the order is
        free and is the policy layer's hook. Defaults to sorted block id.

    Unit bits are MARGINAL: bits_at_tier_k(b) - bits_at_tier_{k-1}(b). Units
    with zero or negative marginal (a block that stopped growing, or is absent
    at this tier) are dropped, so the stream contains exactly the bits the
    client would have sent under the shell path -- verified by a self-test.
    """
    from .transmission import BlockStreamPlan

    per_tier = [per_block_bits_of_config(m, bytes_per_param) for m in tier_models]
    units = []
    for k, cur in enumerate(per_tier):
        prev = per_tier[k - 1] if k > 0 else {}
        marginal = {
            b: cur[b] - prev.get(b, 0) for b in cur if cur[b] - prev.get(b, 0) > 0
        }
        order = (block_order or {}).get(k)
        if order is not None:
            keys = order
        elif rng is not None:
            # Randomised within-shell order. The DEFAULT (sorted by block id) has
            # a systematic bias the mechanism itself creates: the deadline always
            # cuts the same high-index blocks of every shell, so they are
            # chronically under-trained. Shuffling costs nothing (blocks inside a
            # shell are independent -- no prefix semantics) and measures what that
            # bias is worth before a learned ordering policy is built on top.
            keys = sorted(marginal, key=str)
            rng.shuffle(keys)
        else:
            keys = sorted(marginal, key=str)
        for b in keys:
            if b in marginal:
                units.append((k, b, marginal[b]))
    return BlockStreamPlan(units, tier_configs)


class ShardedUpdate:
    """Aggregation view of a trained XL update restricted to delivered units.

    Exposes exactly what `add_subnet` consumes (avg_weight, state_dict(),
    model_index), with state_dict filtered to the delivered blocks' keys —
    convs and bns alike (prefix match). Stem/classifier keys are excluded:
    they are tiny and stay covered by the shell path.
    """

    def __init__(self, client_model, delivered_block_ids, exclude_block_ids=None,
                 include_other=False):
        """delivered_block_ids: blocks to keep (None = keep all blocks).

        exclude_block_ids / include_other added for the block stream
        (DIRECTION §3.1c). A stream client can produce TWO updates at once --
        a tier-k subnet prefix plus blocks that outran it at a larger width --
        and those two overlap on the prefix's coordinates. Emitting both
        unfiltered would count the client's value TWICE on the overlap: same
        value, so the weighted mean is unchanged, but the client's INFLUENCE on
        those coordinates doubles relative to everyone else. That is the exact
        class of silent mis-weighting the MaxNet beta-travels-with-the-slot bug
        already cost this project (§3.3), so the two parts are made disjoint by
        block instead: the prefix excludes the leftover blocks, the leftovers
        carry them at their own level. Both are plain key filters -- no channel
        slice arithmetic, no BN mismatch (sharding.py's granularity argument).

        include_other keeps stem/classifier keys, which the prefix part needs
        (they are part of a valid subnet) and the leftover parts must not
        duplicate.
        """
        self._client_model = client_model
        self._units = None if delivered_block_ids is None else set(delivered_block_ids)
        self._excluded = set(exclude_block_ids or ())
        self._include_other = include_other
        self.avg_weight = client_model.avg_weight
        self.model_index = client_model.model_index
        self.model_config = client_model.model_config

    def state_dict(self):
        full = self._client_model.state_dict()
        kept = {}
        for key in full:
            parts = key.split(".")
            if parts[0] != "blocks":
                if self._include_other:
                    kept[key] = full[key]
                continue
            supernet_block = self.model_index["blocks"][int(parts[1])]
            if supernet_block in self._excluded:
                continue
            if self._units is None or supernet_block in self._units:
                kept[key] = full[key]
        return kept

    def set_avg_wt(self, a):
        self.avg_weight = a
