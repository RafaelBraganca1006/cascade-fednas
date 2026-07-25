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
from policies import base, greedy as greedy_mod, lyapunov as lyap_mod  # noqa: E402
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


def test_policies_gated_by_lambda():
    for ctor in (policies.LyapunovPolicy, policies.ContextualBanditPolicy):
        try:
            ctor()  # no lambda -> must raise NotImplementedError, not silently work
            raise AssertionError(f"{ctor} should require lambda")
        except NotImplementedError:
            pass
    print("PASS lyapunov/bandit correctly refuse to run without lambda (E3-gated)")


def test_bandit_is_still_a_structural_stub():
    bandit = policies.ContextualBanditPolicy(lam=0.1)
    ctx = base.AssignmentContext(round_num=0, client_idx=0, tier_options=_fake_tiers())
    try:
        bandit.assign(ctx)
        raise AssertionError("bandit.assign should still raise NotImplementedError")
    except NotImplementedError:
        pass
    print("PASS bandit remains a structural stub (E6, not yet implemented)")


def _fake_debt(rng, tier_ids=("0", "1", "2", "3")):
    return {tid: rng.uniform(0.0, 100.0) for tid in tier_ids}


def test_lyapunov_degenerates_to_greedy_at_lambda_zero():
    """DIRECTION §3.2: Lyapunov(lambda=0) must exactly reduce to greedy's
    Lemma 1 (always the max tier), for ANY debt distribution -- the
    lambda=0 optimality proof made executable across the policy boundary,
    not just within greedy alone."""
    greedy = greedy_mod.AssignMaxPolicy()
    lyap = lyap_mod.LyapunovPolicy(lam=0.0)
    rng = random.Random(1)
    for trial in range(200):
        tiers = _fake_tiers()
        rng.shuffle(tiers)
        debt = _fake_debt(rng)
        ctx = base.AssignmentContext(
            round_num=trial, client_idx=trial % 20, tier_options=tiers,
            debt_by_tier=debt,
        )
        g = greedy.assign(ctx)
        l = lyap.assign(ctx)
        assert g.tier_id == l.tier_id == "3", (trial, g.tier_id, l.tier_id)
    print("PASS Lyapunov(lambda=0) degenerates to greedy across 200 randomized debts")


def test_lyapunov_protects_s_when_its_debt_dominates():
    """Equal shell sizes (10 Mbit each) isolate the debt/lambda trade-off.
    S starved (debt=100) while M/L/XL just served (debt=0.1) and lambda is
    large (0.9): value(S alone) = 100*10e6 = 1e9, dwarfing any value(i>0)
    (capped near 100e6 by the same S term, now discounted, plus <=3*1e6 from
    the other shells) -- Lyapunov must stay at S rather than risk it."""
    tiers = [
        base.TierOption(tier_id=str(i), config={}, payload_bits=(i + 1) * 10e6)
        for i in range(4)
    ]
    debt = {"0": 100.0, "1": 0.1, "2": 0.1, "3": 0.1}
    ctx = base.AssignmentContext(
        round_num=0, client_idx=0, tier_options=tiers, debt_by_tier=debt,
    )
    chosen = lyap_mod.LyapunovPolicy(lam=0.9).assign(ctx)
    assert chosen.tier_id == "0", chosen
    print("PASS Lyapunov protects a starved S tier when lambda is large")


def test_lyapunov_reaches_high_when_upper_debt_dominates():
    """Mirror scenario: XL starved (debt=100), everything else just served,
    lambda small (0.1) -- XL's debt dwarfs the S-penalty, Lyapunov reaches
    for the max tier exactly like greedy would."""
    tiers = [
        base.TierOption(tier_id=str(i), config={}, payload_bits=(i + 1) * 10e6)
        for i in range(4)
    ]
    debt = {"0": 0.1, "1": 0.1, "2": 0.1, "3": 100.0}
    ctx = base.AssignmentContext(
        round_num=0, client_idx=0, tier_options=tiers, debt_by_tier=debt,
    )
    chosen = lyap_mod.LyapunovPolicy(lam=0.1).assign(ctx)
    assert chosen.tier_id == "3", chosen
    print("PASS Lyapunov reaches for a starved XL tier despite the S-penalty")


def test_lyapunov_missing_debt_raises():
    policy = lyap_mod.LyapunovPolicy(lam=0.1)
    ctx = base.AssignmentContext(
        round_num=0, client_idx=0, tier_options=_fake_tiers(), debt_by_tier=None,
    )
    try:
        policy.assign(ctx)
        raise AssertionError("expected ValueError on missing debt_by_tier")
    except ValueError:
        print("PASS Lyapunov raises on missing debt_by_tier")


def test_lyapunov_empty_tiers_raises():
    policy = lyap_mod.LyapunovPolicy(lam=0.1)
    ctx = base.AssignmentContext(
        round_num=0, client_idx=0, tier_options=[], debt_by_tier={},
    )
    try:
        policy.assign(ctx)
        raise AssertionError("expected ValueError on empty tier_options")
    except ValueError:
        print("PASS Lyapunov raises on empty tier_options")


def test_lyapunov_update_is_noop():
    policy = lyap_mod.LyapunovPolicy(lam=0.1)
    fb = base.DeliveryFeedback(
        round_num=0, client_idx=0, assigned_tier="3", delivered_tier="1",
        native_context=False, native_bits=0.0, cross_context_bits=5e6,
    )
    policy.update(fb)  # must not raise
    print("PASS Lyapunov update() is a no-op (stateless; debt lives in the trainer)")


if __name__ == "__main__":
    test_greedy_always_picks_max_ignoring_context()
    test_greedy_single_tier_edge_case()
    test_greedy_empty_tiers_raises()
    test_greedy_update_is_noop()
    test_policies_registry_has_all_three()
    test_policies_gated_by_lambda()
    test_bandit_is_still_a_structural_stub()
    test_lyapunov_degenerates_to_greedy_at_lambda_zero()
    test_lyapunov_protects_s_when_its_debt_dominates()
    test_lyapunov_reaches_high_when_upper_debt_dominates()
    test_lyapunov_missing_debt_raises()
    test_lyapunov_empty_tiers_raises()
    test_lyapunov_update_is_noop()
    print("ALL PASS")
