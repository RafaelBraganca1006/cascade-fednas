"""Self-test for the assignment-policy package (DIRECTION.md §3.2). Runs
without torch:

    python3 fedml_api/standalone/superfednas/comm_sim/selftest_policy.py

Greedy's test is Lemma 1 made executable: under lambda=0, greedy must
always pick the max tier, regardless of bandwidth/debt/data-utility context
-- if this ever fails, either the lemma's premise broke or the code drifted
from it.
"""

import os
import random
import sys

# The policies/ package has no torch dependency (abc/dataclasses/typing
# only), so -- unlike other selftest_*.py files in this dir, which manually
# load individual files by path to dodge torch imports elsewhere in
# comm_sim -- it's simplest and most faithful to import it as a real
# package, so its internal relative imports (`from .base import ...`)
# resolve normally instead of fighting module identity across ad-hoc loads.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policies import base, greedy as greedy_mod  # noqa: E402
import policies  # noqa: E402


def _fake_tiers():
    # deliberately unsorted-by-value payloads to make "max" nontrivial to get
    # right by accident; ids don't correlate with size.
    raw = [("2", 40e6), ("0", 5e6), ("3", 150e6), ("1", 20e6)]
    return [
        base.TierOption(tier_id=t, config={"e": t}, payload_bits=b) for t, b in raw
    ]


def test_greedy_always_picks_max_ignoring_context():
    policy = greedy_mod.AssignMaxPolicy()
    rng = random.Random(0)
    for trial in range(200):
        tiers = _fake_tiers()
        rng.shuffle(tiers)  # order in tier_options must not matter
        ctx = base.AssignmentContext(
            round_num=trial,
            client_idx=trial % 20,
            tier_options=tiers,
            # randomize every context field greedy is supposed to ignore
            bw_estimate_bps=rng.choice([None, 0.0, 1e6, 50e6]),
            upload_window_s=rng.choice([None, 0.0, 20.0, 120.0]),
            debt_by_tier=rng.choice([None, {}, {"3": 999.0, "0": -1.0}]),
            data_utility=rng.choice([None, {"loss": 2.3}]),
        )
        chosen = policy.assign(ctx)
        assert chosen.tier_id == "3" and chosen.payload_bits == 150e6, (
            trial, chosen,
        )
    print("PASS greedy always picks the max tier (Lemma 1), 200 randomized contexts")


def test_greedy_single_tier_edge_case():
    policy = greedy_mod.AssignMaxPolicy()
    only = base.TierOption(tier_id="0", config={}, payload_bits=5e6)
    ctx = base.AssignmentContext(round_num=0, client_idx=0, tier_options=[only])
    assert policy.assign(ctx) is only
    print("PASS greedy single-tier edge case")


def test_greedy_empty_tiers_raises():
    policy = greedy_mod.AssignMaxPolicy()
    ctx = base.AssignmentContext(round_num=0, client_idx=0, tier_options=[])
    try:
        policy.assign(ctx)
        raise AssertionError("expected ValueError on empty tier_options")
    except ValueError:
        print("PASS greedy raises on empty tier_options")


def test_greedy_update_is_noop():
    policy = greedy_mod.AssignMaxPolicy()
    fb = base.DeliveryFeedback(
        round_num=0, client_idx=0, assigned_tier="3", delivered_tier="0",
        native_context=False, native_bits=0.0, cross_context_bits=5e6,
    )
    policy.update(fb)  # must not raise
    print("PASS greedy update() is a no-op (stateless)")


def test_policies_registry_has_all_three():
    assert set(policies.POLICIES) == {"greedy", "lyapunov", "bandit"}, policies.POLICIES
    assert policies.POLICIES["greedy"] is greedy_mod.AssignMaxPolicy
    print("PASS POLICIES registry has all three: greedy, lyapunov, bandit")


def test_lyapunov_and_bandit_are_structural_stubs():
    for ctor in (policies.LyapunovPolicy, policies.ContextualBanditPolicy):
        try:
            ctor()  # no lambda -> must raise NotImplementedError, not silently work
            raise AssertionError(f"{ctor} should require lambda")
        except NotImplementedError:
            pass
    print("PASS lyapunov/bandit correctly refuse to run without lambda (E3-gated)")


if __name__ == "__main__":
    test_greedy_always_picks_max_ignoring_context()
    test_greedy_single_tier_edge_case()
    test_greedy_empty_tiers_raises()
    test_greedy_update_is_noop()
    test_policies_registry_has_all_three()
    test_lyapunov_and_bandit_are_structural_stubs()
    print("ALL PASS")
