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


class ShardedUpdate:
    """Aggregation view of a trained XL update restricted to delivered units.

    Exposes exactly what `add_subnet` consumes (avg_weight, state_dict(),
    model_index), with state_dict filtered to the delivered blocks' keys —
    convs and bns alike (prefix match). Stem/classifier keys are excluded:
    they are tiny and stay covered by the shell path.
    """

    def __init__(self, client_model, delivered_block_ids):
        self._client_model = client_model
        self._units = set(delivered_block_ids)
        self.avg_weight = client_model.avg_weight
        self.model_index = client_model.model_index
        self.model_config = client_model.model_config

    def state_dict(self):
        full = self._client_model.state_dict()
        kept = {}
        for key in full:
            parts = key.split(".")
            if parts[0] != "blocks":
                continue
            supernet_block = self.model_index["blocks"][int(parts[1])]
            if supernet_block in self._units:
                kept[key] = full[key]
        return kept

    def set_avg_wt(self, a):
        self.avg_weight = a
