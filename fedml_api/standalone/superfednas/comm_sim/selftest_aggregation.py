#!/usr/bin/env python3
"""Self-tests for the aggregation weight rules (comm_sim/aggregation.py).

WHY THIS FILE MATTERS MORE THAN THE OTHER SELF-TESTS
====================================================
Until now the aggregation path had NO test coverage, and it is where every
silent bug in this project came from -- a run completes, the accuracy looks
plausible, and the wrong quantity was measured. The specific failures this
suite is built against, all of which really happened:

  - "beta travels with the slot": a max-slot weight applied to whatever prefix
    arrived, because the calendar was set before delivery was known.
  - a client's update counted twice on overlapping coordinates when the block
    stream emitted a prefix and a leftover that were not disjoint.
  - a weight rule that quietly changes the round's total mass, which rescales
    the aggregate and is indistinguishable from a real effect in a results
    table.

Torch-free and runs in milliseconds, so it belongs to the fast suite next to
selftest_transmission.py and the rest.

    python3 selftest_aggregation.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comm_sim.aggregation import (  # noqa: E402
    Delivery,
    cascade_weights,
    completeness_weights,
    slot_weights,
)

TOL = 1e-9


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def maxnet_slots(n, beta, max_idx=0):
    """Reproduce maxnet_cos_all_subnet's output: beta to the max slot, the
    remainder split evenly among the rest."""
    other = (1.0 - beta) / (n - 1)
    return [beta if i == max_idx else other for i in range(n)]


def make_round(n=8, beta=0.9, delivered_tiers=None, num_tiers=4,
               completeness=None, coverage=None, counts=None, rounds_seen=150):
    slots = maxnet_slots(n, beta)
    if delivered_tiers is None:
        delivered_tiers = [num_tiers - 1] * n
    if completeness is None:
        completeness = [1.0] * n
    return [
        Delivery(
            client_idx=i,
            slot_weight=slots[i],
            delivered_tier=delivered_tiers[i],
            num_tiers=num_tiers,
            completeness=completeness[i],
            region_coverage=None if coverage is None else coverage[i],
            delivery_count=None if counts is None else counts[i],
            round_count=None if counts is None else rounds_seen,
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------
def test_mass_conservation():
    """Every rule except the shipped one must preserve the round's total weight.

    A rule that changes the mass rescales the aggregate. The resulting accuracy
    move would be an artefact of scaling, not of aggregation, and would be
    indistinguishable from a real effect in the results table.
    """
    for tiers in ([3] * 8, [0, 1, 2, 3, 3, 1, 0, 2], [-1, 0, 0, 1, 1, 2, 2, 3]):
        d = make_round(delivered_tiers=tiers,
                       coverage=[0.2, 0.9, 0.5, 0.1, 0.8, 0.3, 0.6, 0.4],
                       counts=[1, 90, 40, 2, 80, 20, 60, 30])
        target = sum(slot_weights(d))
        got = sum(cascade_weights(d))
        assert approx(got, target), (
            "cascade changed the round mass: %.6f vs %.6f" % (got, target))
    print("PASS mass conservation: cascade preserves the round's total weight")


def test_degenerates_to_maxnet():
    """With complete delivery and no observed-state factors, cascade == MaxNet.

    This is the reachability property that makes a loss attributable: MaxNet is
    a member of the rule's class, so if cascade loses, it lost on the rule and
    not because the action space changed. Same discipline as `cap = XL`
    reproducing blind sampling bit-identically.
    """
    n, beta = 8, 0.9
    # exactly one client delivers the max tier -- the slot holder -- and
    # everyone else delivers a lower tier, which is what a clean round looks
    # like when nothing is truncated.
    tiers = [3] + [1] * (n - 1)
    d = make_round(n=n, beta=beta, delivered_tiers=tiers)
    got = cascade_weights(d)
    want = slot_weights(d)
    for i, (g, w) in enumerate(zip(got, want)):
        assert approx(g, w), "client %d: %.6f != %.6f" % (i, g, w)
    print("PASS degeneracy: complete delivery reproduces the MaxNet slot weights")


def test_beta_is_a_budget_not_a_bonus():
    """When several clients deliver the max tier, the max mass is SPLIT.

    Applying beta once per max-tier deliverer would inflate the top of the
    ladder by the number of clients that got through, which is exactly the
    "stacked multiplicative weights" failure the module exists to avoid. The
    round's top-of-ladder influence must not depend on how many made it.
    """
    n, beta = 8, 0.9
    one = make_round(n=n, beta=beta, delivered_tiers=[3] + [1] * (n - 1))
    three = make_round(n=n, beta=beta, delivered_tiers=[3, 3, 3] + [1] * (n - 3))

    w1 = cascade_weights(one)
    w3 = cascade_weights(three)
    assert approx(sum(w1), sum(w3)), "mass differs between rounds"
    top1 = w1[0]
    top3 = sum(w3[:3])
    assert approx(top1, top3), (
        "top-of-ladder mass changed with the number of deliverers: "
        "%.6f vs %.6f" % (top1, top3))
    assert w3[0] < w1[0], "individual max weight should shrink when shared"
    print("PASS beta is a budget: max-tier mass is split, not repeated")


def test_no_max_delivered_shares_evenly():
    """When nobody delivers the top tier, the earmarked mass is not a windfall.

    Measured reality at the canonical config: the complete max tier arrives
    under 2% of the time. If the rule handed beta to whoever happened to be
    closest, one arbitrary client would dominate most rounds.
    """
    d = make_round(delivered_tiers=[0, 1, 2, 1, 0, 2, 1, 0])
    w = cascade_weights(d)
    assert approx(max(w), min(w)), "weights should be equal when nobody tops out"
    assert approx(sum(w), sum(slot_weights(d)))
    print("PASS no max delivered: the earmarked mass is shared, not handed out")


def test_coverage_favours_starved_regions():
    """A delivery into a starved region outweighs one into a covered region."""
    tiers = [2] * 8
    starved = make_round(delivered_tiers=tiers,
                         coverage=[0.0] + [0.9] * 7)
    w = cascade_weights(starved)
    assert w[0] > w[1], "starved region should weigh more (%.4f vs %.4f)" % (
        w[0], w[1])
    assert approx(sum(w), sum(slot_weights(starved)))
    print("PASS coverage: a starved region outweighs a well-covered one")


def test_inverse_rate_lifts_the_rarely_heard():
    """A client that rarely gets through counts for more when it does.

    The Horvitz-Thompson argument: its delivery stands in for the ones the
    channel destroyed. Measured target from 104207 -- poorly connected clients
    complete a tier 44 times against the well-connected group's 322.
    """
    tiers = [2] * 8
    d = make_round(delivered_tiers=tiers, counts=[2] + [90] * 7)
    w = cascade_weights(d)
    assert w[0] > w[1], "rare deliverer should weigh more (%.4f vs %.4f)" % (
        w[0], w[1])
    assert approx(sum(w), sum(slot_weights(d)))
    print("PASS inverse rate: a rarely-heard client counts for more when heard")


def test_clip_bounds_the_rare_client():
    """HT variance explodes as p -> 0, so the weight must be clipped.

    Without the clip a client that has never delivered would take almost the
    whole round the first time it gets through.
    """
    tiers = [2] * 8
    d = make_round(delivered_tiers=tiers, counts=[0] + [140] * 7,
                   rounds_seen=150)
    w = cascade_weights(d, w_max=4.0)
    mean_slot = sum(slot_weights(d)) / len(d)
    # after renormalisation the clip is not a hard bound on the final value, so
    # assert the property that matters: no client takes an absurd share.
    assert w[0] < 0.5 * sum(w), (
        "a never-delivering client took %.1f%% of the round"
        % (100 * w[0] / sum(w)))
    assert mean_slot > 0
    print("PASS clip: a never-delivering client cannot take over the round")


def test_completeness_rule_is_unchanged():
    """The shipped rule must keep behaving exactly as the archived runs saw it.

    It deliberately does NOT renormalise -- the round's mass shrinks under
    truncation. That is the shipped semantics and changing it would silently
    invalidate every archived --maxnet_delivery_aware run.
    """
    d = make_round(delivered_tiers=[3] + [1] * 7,
                   completeness=[0.5] + [1.0] * 7)
    w = completeness_weights(d)
    assert approx(w[0], 0.9 * 0.5), "max slot should be halved"
    assert sum(w) < sum(slot_weights(d)), "shipped rule loses mass by design"
    print("PASS shipped rule: completeness scaling unchanged, mass still shrinks")


def test_empty_and_degenerate_rounds():
    """Nothing crashes on an empty round or an all-zero round."""
    assert cascade_weights([]) == []
    d = [Delivery(0, 0.0, -1, 4), Delivery(1, 0.0, -1, 4)]
    w = cascade_weights(d)
    assert len(w) == 2 and all(x == 0.0 for x in w), w
    print("PASS degenerate rounds: empty and all-zero handled without crashing")


if __name__ == "__main__":
    test_mass_conservation()
    test_degenerates_to_maxnet()
    test_beta_is_a_budget_not_a_bonus()
    test_no_max_delivered_shares_evenly()
    test_coverage_favours_starved_regions()
    test_inverse_rate_lifts_the_rarely_heard()
    test_clip_bounds_the_rare_client()
    test_completeness_rule_is_unchanged()
    test_empty_and_degenerate_rounds()
    print("ALL AGGREGATION SELF-TESTS PASSED")
