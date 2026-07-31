"""Self-tests for the unified block-granular stream (DIRECTION §3.1c).

    python3 fedml_api/standalone/superfednas/comm_sim/selftest_block_stream.py

Pure-python, no torch. The load-bearing one is
``test_coarse_granularity_reproduces_nested``: with one block per shell the
stream MUST behave exactly like the shell path. That makes "the old nested mode
is a special case of the new one" true rather than asserted, so any regression
in the 150-round arms is attributable to the new granularity and not to a
rewrite -- the same discipline as cap=XL reproducing blind.
"""

import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tx = _load("transmission", os.path.join(_here, "transmission.py"))

MB = 8e6  # bits per MB
# The E6b ladder, in MB: S=41.6 M=98.6 L=137.6 XL=286.9 -> shells 41.6/57/39/149.3
SHELL_MB = [41.6, 57.0, 39.0, 149.3]
CFGS = [{"t": 0}, {"t": 1}, {"t": 2}, {"t": 3}]


def stream(blocks_per_shell=4, shell_mb=None):
    """A plan with `blocks_per_shell` equal blocks per shell."""
    shell_mb = shell_mb or SHELL_MB
    units = []
    for k, mb in enumerate(shell_mb):
        for b in range(blocks_per_shell):
            units.append((k, b, mb * MB / blocks_per_shell))
    return tx.BlockStreamPlan(units, CFGS)


def shell_plan(shell_mb=None):
    shell_mb = shell_mb or SHELL_MB
    cum, tiers = 0.0, []
    for mb in shell_mb:
        cum += mb
        tiers.append((cum * MB, {"t": len(tiers)}))
    return tx.TransmissionPlan(tiers, train_tier=len(tiers) - 1)


def test_totals_match_shell_path():
    p, s = stream(), shell_plan()
    assert abs(p.total_bits() - s.total_bits()) < 1.0, (p.total_bits(), s.total_bits())
    for k in range(4):
        assert abs(p.shell_bits(k) - s.shell_bits(k)) < 1.0, k
    print("PASS block units sum exactly to the shell path's payload")


def test_coarse_granularity_reproduces_nested():
    """One block per shell => identical classification to truncate()."""
    for budget_mb in [0, 10, 41.5, 41.6, 60, 98.6, 100, 137.6, 200, 286.9, 400]:
        budget = budget_mb * MB
        a = tx.truncate_stream(stream(1), budget, 1e9, 40.0)
        b = tx.truncate(shell_plan(), budget, 1e9, 40.0)
        assert a.delivered_tier == b.delivered_tier, (
            budget_mb, a.delivered_tier, b.delivered_tier)
        assert abs(a.bits_delivered - b.bits_delivered) < 1.0, budget_mb
        assert a.completed_plan == b.completed_plan, budget_mb
        assert not a.leftover_blocks, (budget_mb, a.leftover_blocks)
    print("PASS coarse granularity == the frozen nested path (11 budgets)")


def test_sub_floor_client_now_delivers():
    """THE point of the mode: a budget below the smallest tier used to deliver
    zero. Measured motivation: 34.8% of slots, average budget 16.6 MB vs a
    41.6 MB smallest tier."""
    budget = 16.6 * MB
    old = tx.truncate(shell_plan(), budget, 1e9, 40.0)
    assert old.delivered_tier == -1 and old.delivers_nothing()
    assert old.bits_delivered == 0.0

    new = tx.truncate_stream(stream(8), budget, 1e9, 40.0)
    assert new.delivered_tier == -1, "no complete tier expected at 16.6 MB"
    assert not new.delivers_nothing(), "stream must still deliver whole blocks"
    assert new.leftover_blocks, new.leftover_blocks
    assert new.bits_delivered > 0
    print("PASS sub-floor client: shell path 0 MB -> stream %.1f MB usable"
          % (new.bits_delivered / MB))


def test_waste_is_at_most_one_block():
    """Max waste drops from one whole SHELL to one cut BLOCK.

    The bound is the LARGEST unit, not the first: shells are very unevenly
    sized in this ladder (S=41.6 MB but the top shell is 149.3 MB), so at 8
    blocks per shell a top-shell block is 18.7 MB against S's 5.2 MB. Worth
    recording for the thesis -- the waste bound is set by the coarsest block in
    the ladder, which is an argument for splitting the top shell more finely
    than the bottom one rather than using a uniform block count.
    """
    p = stream(8)
    largest_unit = max(bits for _, _, bits in p.units)
    largest_shell = max(p.shell_bits(k) for k in range(p.num_shells))
    worst = 0.0
    for i in range(1, 400):
        r = tx.truncate_stream(p, i * 1.7 * MB, 1e9, 40.0)
        if r.completed_plan:
            continue
        worst = max(worst, r.bits_attempted - r.bits_delivered)
    assert worst <= largest_unit + 1.0, (worst / MB, largest_unit / MB)
    assert worst < largest_shell, "no better than the shell path"
    print("PASS max waste %.2f MB <= largest block %.2f MB "
          "(shell path's bound: %.2f MB)"
          % (worst / MB, largest_unit / MB, largest_shell / MB))


def test_leftovers_only_above_the_prefix():
    """A block reported as leftover must be strictly above the global tier, and
    every unit up to its level must actually have arrived."""
    for budget_mb in [30, 55, 80, 120, 160, 250]:
        p = stream(6)
        r = tx.truncate_stream(p, budget_mb * MB, 1e9, 40.0)
        for b, lv in r.leftover_blocks.items():
            assert lv > r.delivered_tier, (budget_mb, b, lv, r.delivered_tier)
        for b, lv in r.block_levels.items():
            assert lv >= r.delivered_tier, (budget_mb, b, lv, r.delivered_tier)
    print("PASS leftovers strictly above the delivered prefix, no gaps")


def test_depth_elastic_blocks():
    """A block absent from shallow tiers must not break the prefix rule."""
    units = [(0, "a", 10 * MB), (1, "a", 10 * MB), (1, "b", 10 * MB),
             (2, "a", 10 * MB), (2, "b", 10 * MB), (2, "c", 10 * MB)]
    p = tx.BlockStreamPlan(units, CFGS[:3])
    r = tx.truncate_stream(p, 30 * MB, 1e9, 40.0)
    assert r.delivered_tier == 1, r.delivered_tier   # shells 0 and 1 complete
    r2 = tx.truncate_stream(p, 15 * MB, 1e9, 40.0)
    assert r2.delivered_tier == 0, r2.delivered_tier  # only shell 0 complete
    assert r2.leftover_blocks == {}, r2.leftover_blocks
    print("PASS depth-elastic blocks (absent from shallow tiers) handled")


def test_mixed_block_id_types():
    """Block ids are ints EXCEPT the OTHER_BLOCK sentinel (stem/classifier), so
    every collection of them must survive sorting/serialisation.

    Added after job 103854 crashed on `sorted(leftover_blocks.items())` with
    "'<' not supported between instances of 'str' and 'int'". The earlier tests
    all used integer ids only, so the mixed case was never exercised -- and
    fixed order happened to place OTHER_BLOCK last in its shell, hiding it until
    --block_order random made it likely. Anything that orders block ids must use
    key=str.
    """
    units = []
    for k, mb in enumerate(SHELL_MB):
        for b in [0, 1, tx.OTHER_BLOCK, 2]:
            units.append((k, b, mb * MB / 4))
    p = tx.BlockStreamPlan(units, CFGS)
    for budget_mb in [20, 45, 70, 110, 150, 200, 260]:
        r = tx.truncate_stream(p, budget_mb * MB, 1e9, 40.0)
        # the exact call that crashed in production
        sorted(r.leftover_blocks.items(), key=lambda kv: str(kv[0]))
        sorted(r.block_levels.items(), key=lambda kv: str(kv[0]))
        assert all(lv > r.delivered_tier for lv in r.leftover_blocks.values())
    seen = {b for _, b, _ in p.units}
    assert tx.OTHER_BLOCK in seen and 0 in seen, seen
    print("PASS mixed int / OTHER_BLOCK ids sort and classify without crashing")


def test_zero_budget():
    r = tx.truncate_stream(stream(4), 0.0, 1e9, 40.0)
    assert r.delivered_tier == -1 and r.delivers_nothing()
    assert r.bits_delivered == 0.0 and not r.leftover_blocks
    print("PASS zero budget delivers nothing, no crash")


# ---------------------------------------------------------------------------
# Cohort-complementary streaming (DIRECTION 3.1d, --stream_pivot)
# ---------------------------------------------------------------------------

# sharding.py does `from .transmission import ...`, so it needs a parent
# package to resolve against. Register a synthetic one holding the already
# loaded transmission module -- keeps this file torch-free and importable
# standalone, same as the rest of the self-tests.
import sys  # noqa: E402
import types  # noqa: E402

_pkg = types.ModuleType("_commsim")
_pkg.__path__ = [_here]
sys.modules["_commsim"] = _pkg
sys.modules["_commsim.transmission"] = tx
sh = _load("_commsim.sharding", os.path.join(_here, "sharding.py"))


class _Tensor:
    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n


class _StubModel:
    """Minimal stand-in for a client model: build_block_stream only ever reads
    state_dict() and model_index, so no torch is needed to test the ORDERING."""

    def __init__(self, n_blocks, per_block_numel, other_numel=8):
        self._sd = {"stem.weight": _Tensor(other_numel)}
        for b in range(n_blocks):
            self._sd["blocks.%d.conv.weight" % b] = _Tensor(per_block_numel)
        self.model_index = {"blocks": {b: b for b in range(n_blocks)}}
        self.avg_weight = 1.0        # ShardedUpdate copies these off the model
        self.model_config = {"d": [0], "e": 0.1}

    def state_dict(self):
        return self._sd


def _ladder(n_blocks=4, widths=(2, 4, 6, 12)):
    """One model per tier; block payload grows with the tier (width nesting)."""
    return [_StubModel(n_blocks, w) for w in widths], list(CFGS)


def test_pivot_default_is_bit_identical():
    """The load-bearing one: pivot=None must reproduce the shell-major stream
    EXACTLY, or every archived nested_blocks run stops being comparable."""
    models, cfgs = _ladder()
    base = sh.build_block_stream(models, cfgs)
    for off in (0, 1, 7):
        same = sh.build_block_stream(models, cfgs, pivot=None, rotation_offset=off)
        assert same.units == base.units, "pivot=None must ignore rotation_offset"
    assert sh.build_block_stream(models, cfgs, pivot=4).units == base.units, \
        "pivot == n_shells is pure shell-major, i.e. the default"
    print("PASS pivot default (and pivot=n_shells) is bit-identical to shell-major")


def test_pivot_preserves_total_bits():
    """Reordering must never change WHAT is sent, only the order -- otherwise
    the byte accounting stops being comparable across arms."""
    models, cfgs = _ladder()
    base = sh.build_block_stream(models, cfgs)
    for pivot in (0, 1, 2, 3):
        for off in (0, 2, 5):
            p = sh.build_block_stream(
                models, cfgs, pivot=pivot, rotation_offset=off
            )
            assert abs(p.total_bits() - base.total_bits()) < 1e-6
            assert sorted(p.units, key=str) == sorted(base.units, key=str)
    print("PASS every pivot/offset is a permutation of the same units")


def test_block_major_tail_never_skips_within_a_block():
    """A block's shells are nested in WIDTH, so its outer channels are useless
    without its inner ones. The tail may reorder ACROSS blocks, never within."""
    models, cfgs = _ladder()
    p = sh.build_block_stream(models, cfgs, pivot=1, rotation_offset=2)
    seen = {}
    for k, b, _ in p.units:
        if k == 0:
            continue
        prev = seen.get(b)
        assert prev is None or k > prev, "shells within a block must ascend"
        seen[b] = k
    print("PASS block-major tail keeps each block's shells in ascending order")


def test_rotation_makes_the_cohort_cover_the_top():
    """The point of the mechanism: with a budget that reaches only a fraction
    of the top shell, shell-major has EVERY client stall in the same place, so
    the cohort covers nothing new; rotation spreads the stalls and the union
    covers the top."""
    models, cfgs = _ladder(n_blocks=4)
    top = len(cfgs) - 1
    budget = sh.build_block_stream(models, cfgs).total_bits() * 0.55

    def union_top(pivot, offsets):
        covered = set()
        for off in offsets:
            p = sh.build_block_stream(
                models, cfgs, pivot=pivot, rotation_offset=off
            )
            r = tx.truncate_stream(p, budget, 1e9, 40.0)
            covered |= {b for b, lv in r.block_levels.items() if lv >= top}
        return covered

    offsets = [0, 1, 2, 3]
    shell_major = union_top(None, offsets)
    rotated = union_top(1, offsets)
    assert len(rotated) > len(shell_major), (shell_major, rotated)
    print("PASS rotation lifts the cohort's top-shell coverage %d -> %d blocks"
          % (len(shell_major), len(rotated)))


def test_other_block_never_lands_in_the_tail():
    """stem/classifier must stay in the head: ShardedUpdate drops non-`blocks`
    keys from collective updates, so an OTHER_BLOCK unit arriving in the tail
    would be charged as delivered bytes and never aggregated."""
    models, cfgs = _ladder()
    # give the stem a marginal at every tier, so it WOULD be a tail unit
    for i, m in enumerate(models):
        m._sd["stem.weight"] = _Tensor(4 * (i + 1))
    p = sh.build_block_stream(models, cfgs, pivot=1, rotation_offset=3)
    tail_start = max(i for i, (k, _b, _s) in enumerate(p.units) if k == 0) + 1
    other_pos = [i for i, (_k, b, _s) in enumerate(p.units) if b is tx.OTHER_BLOCK]
    assert other_pos, "stem should still be in the stream"
    n_blocks = len({b for _k, b, _s in p.units if b is not tx.OTHER_BLOCK})
    # every OTHER_BLOCK unit comes before the first real tail block completes
    first_tail_block = next(b for k, b, _s in p.units[tail_start:]
                            if b is not tx.OTHER_BLOCK)
    first_tail_pos = next(i for i, (_k, b, _s) in enumerate(p.units)
                          if b is first_tail_block and i >= tail_start)
    assert max(other_pos) < first_tail_pos, (other_pos, first_tail_pos, n_blocks)
    print("PASS OTHER_BLOCK stays in the head, never a block-major tail unit")


def test_rotation_spread_is_even_across_the_cohort():
    """The offsets of a round's K clients must spread over the block space.
    Using raw client indices (the natural but wrong choice) clumps them: a
    sampled set like [12, 4, 18, 0, 9, 5, 3, 10] puts ranks 3/4/5 one block
    apart, covering the same region while others get none."""
    models, cfgs = _ladder(n_blocks=8)
    K = 4

    def first_tail_blocks(spread_offsets):
        seen = []
        for off, spread in spread_offsets:
            p = sh.build_block_stream(models, cfgs, pivot=1,
                                      rotation_offset=off, rotation_spread=spread)
            tail = [b for k, b, _s in p.units if k > 0 and b is not tx.OTHER_BLOCK]
            seen.append(tail[0])
        return seen

    even = first_tail_blocks([(0, (i, K)) for i in range(K)])
    clumped = first_tail_blocks([(i, None) for i in (12, 4, 18, 0)])
    assert len(set(even)) == K, f"rank-spread offsets must be distinct: {even}"
    gaps = sorted((even[(i + 1) % K] - even[i]) % 8 for i in range(K))
    assert min(gaps) >= 1, gaps
    print("PASS cohort offsets spread evenly (%s) vs index-based (%s)"
          % (even, clumped))



def test_sharded_update_exclusion_keeps_stem_and_the_anchor():
    m = _StubModel(n_blocks=4, per_block_numel=6)
    u = sh.ShardedUpdate(m, None, exclude_block_ids=[0, 1], include_other=True)
    keys = set(u.state_dict())
    assert "stem.weight" in keys, "include_other must keep stem/classifier"
    assert "blocks.3.conv.weight" in keys, "anchor block must survive"
    assert not any(k.startswith(("blocks.0.", "blocks.1.")) for k in keys), keys
    print("PASS ShardedUpdate drops the excluded blocks, keeps stem + anchor")


if __name__ == "__main__":
    test_totals_match_shell_path()
    test_coarse_granularity_reproduces_nested()
    test_sub_floor_client_now_delivers()
    test_waste_is_at_most_one_block()
    test_leftovers_only_above_the_prefix()
    test_depth_elastic_blocks()
    test_mixed_block_id_types()
    test_zero_budget()
    test_pivot_default_is_bit_identical()
    test_pivot_preserves_total_bits()
    test_block_major_tail_never_skips_within_a_block()
    test_rotation_makes_the_cohort_cover_the_top()
    test_other_block_never_lands_in_the_tail()
    test_rotation_spread_is_even_across_the_cohort()
    test_sharded_update_exclusion_keeps_stem_and_the_anchor()
    print("ALL PASS")
