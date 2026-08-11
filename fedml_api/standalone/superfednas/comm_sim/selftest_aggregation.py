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
  - a cap applied to the wrong quantity: bounding the TOTAL weight instead of
    the correction cut MaxNet's own max slot and broke degeneracy, which the
    degeneracy test caught on its first run (0.833 against 0.900).

One belief this suite deliberately does NOT encode: that the round's total
weight must be conserved. `_aggregate` computes a per-coordinate weighted mean,
so it is already self-normalising and only RELATIVE weights reach the model. An
earlier version of this file tested mass conservation as if it were a safety
property; test_relative_weights_are_what_matter replaces it and keeps the real
fact visible.

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
    """Reproduce maxnet_cos_all_subnet: beta to the max slot, remainder split."""
    other = (1.0 - beta) / (n - 1)
    return [beta if i == max_idx else other for i in range(n)]


def make_round(n=8, beta=0.9, tiers=None, num_tiers=4, completeness=None,
               freq=None):
    slots = maxnet_slots(n, beta)
    if tiers is None:
        tiers = [num_tiers - 1] * n
    if completeness is None:
        completeness = [1.0] * n
    return [
        Delivery(
            client_idx=i,
            slot_weight=slots[i],
            delivered_tier=tiers[i],
            num_tiers=num_tiers,
            completeness=completeness[i],
            coverage_freq=None if freq is None else freq[i],
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------
def test_degenerates_to_the_base_rule():
    """With every frequency at 1 -- complete delivery -- cascade IS MaxNet.

    The reachability property that makes a loss attributable: the base method is
    a special case of the proposed rule, so a difference comes from the
    correction and not from a changed action space. Same discipline as
    `cap = XL` reproducing blind sampling bit-identically.
    """
    d = make_round(freq=[1.0] * 8)
    for i, (g, w) in enumerate(zip(cascade_weights(d), slot_weights(d))):
        assert approx(g, w), "client %d: %.6f != %.6f" % (i, g, w)
    # and with no frequency supplied at all
    d2 = make_round()
    for g, w in zip(cascade_weights(d2), slot_weights(d2)):
        assert approx(g, w)
    print("PASS degeneracy: complete delivery reproduces the base weights")


def test_rare_region_is_upweighted():
    """A contribution to a region this client rarely reaches counts for more.

    This is the whole correction: if client i reaches region s a fraction
    pi(i,s) of the time, the plain mean over arrivals is biased toward the
    clients that arrive often, and 1/pi cancels it.
    """
    # Frequencies chosen so the cap does not bind (1/0.3 = 3.33 < w_max = 4),
    # which isolates the correction from the clipping tested below. A first
    # version used 0.1 and expected a ratio of 8; the cap correctly cut it to
    # 3.2, so the test was wrong and the code was right.
    d = make_round(tiers=[2] * 8, freq=[0.3] + [0.9] * 7)
    w = cascade_weights(d)
    assert w[0] > w[1], "rare region should weigh more (%.4f vs %.4f)" % (
        w[0], w[1])
    # The slot weights differ, so compare the CORRECTION ratio: 0.9/0.3 = 3.
    ratio = (w[0] / d[0].slot_weight) / (w[1] / d[1].slot_weight)
    assert approx(ratio, 3.0, tol=1e-6), ratio
    print("PASS correction: a rarely-reached region is upweighted by 1/pi")


def test_relative_weights_are_what_matter():
    """Scaling every weight by a constant must not change the aggregate.

    _aggregate computes a per-coordinate weighted MEAN, so it is already
    self-normalising. This test exists to keep that fact visible: an earlier
    version of this module claimed mass conservation was a safety property, and
    it is not. What must be tested is the RELATIVE behaviour.
    """
    d = make_round(tiers=[1, 2, 3, 0, 2, 1, 3, 2],
                   freq=[0.2, 0.5, 0.15, 0.9, 0.4, 0.3, 0.15, 0.6])
    w = cascade_weights(d)
    params = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    mean = sum(a * b for a, b in zip(w, params)) / sum(w)
    scaled = [x * 13.7 for x in w]
    mean2 = sum(a * b for a, b in zip(scaled, params)) / sum(scaled)
    assert approx(mean, mean2, tol=1e-12), (mean, mean2)
    print("PASS scale invariance: only relative weights reach the model")


def test_cap_bounds_the_correction_not_the_slot():
    """The cap must bound the amplification, never MaxNet's own allocation.

    beta = 0.9 makes the max slot legitimately 7x the mean weight by MaxNet's
    design. An earlier version capped the TOTAL weight, which cut the max slot
    to the ceiling and made degeneracy fail (0.833 against 0.900). Caught by the
    degeneracy test on its first run.
    """
    d = make_round(freq=[1.0] * 8)
    w = cascade_weights(d, w_max=4.0)
    assert approx(w[0], 0.9), "the max slot was capped: %.4f" % w[0]
    # a vanishing frequency is bounded by w_max, not unbounded
    d2 = make_round(tiers=[2] * 8, freq=[1e-9] + [1.0] * 7)
    w2 = cascade_weights(d2, w_max=4.0)
    corr = (w2[0] / d2[0].slot_weight) / (w2[1] / d2[1].slot_weight)
    assert corr <= 4.0 + 1e-6, "correction exceeded the cap: %.3f" % corr
    print("PASS cap: bounds the correction, leaves the slot allocation intact")


def test_unbiasedness_in_simulation():
    """The estimator recovers the full-participation mean, which is the claim.

    Simulate a region reached by client A with probability 0.8 and by client B
    with probability 0.2, where the two hold different updates. The plain mean
    over arrivals is pulled toward A; weighting by 1/pi must recover the
    unweighted average of A and B, which is what full delivery would have given.
    """
    import random
    rng = random.Random(0)
    val = {0: 1.0, 1: 5.0}
    pi = {0: 0.8, 1: 0.2}
    truth = (val[0] + val[1]) / 2.0

    plain_num = plain_den = corr_num = corr_den = 0.0
    for _ in range(200000):
        for c in (0, 1):
            if rng.random() < pi[c]:
                plain_num += val[c]
                plain_den += 1.0
                w = 1.0 / pi[c]
                corr_num += w * val[c]
                corr_den += w
    plain = plain_num / plain_den
    corrected = corr_num / corr_den
    assert abs(corrected - truth) < 0.02, corrected
    assert abs(plain - truth) > 0.5, plain
    print("PASS unbiasedness: 1/pi recovers %.2f; the plain mean gives %.2f"
          % (corrected, plain))


def test_completeness_rule_is_unchanged():
    """The shipped rule must behave exactly as the archived runs saw it."""
    d = make_round(tiers=[3] + [1] * 7, completeness=[0.5] + [1.0] * 7)
    w = completeness_weights(d)
    assert approx(w[0], 0.9 * 0.5), "max slot should be halved"
    assert sum(w) < sum(slot_weights(d)), "shipped rule loses mass by design"
    print("PASS shipped rule: completeness scaling unchanged")


def test_empty_and_degenerate_rounds():
    assert cascade_weights([]) == []
    d = [Delivery(0, 0.0, -1, 4), Delivery(1, 0.0, -1, 4)]
    w = cascade_weights(d)
    assert len(w) == 2 and all(x == 0.0 for x in w), w
    print("PASS degenerate rounds: empty and all-zero handled")


if __name__ == "__main__":
    test_degenerates_to_the_base_rule()
    test_rare_region_is_upweighted()
    test_relative_weights_are_what_matter()
    test_cap_bounds_the_correction_not_the_slot()
    test_unbiasedness_in_simulation()
    test_completeness_rule_is_unchanged()
    test_empty_and_degenerate_rounds()
    print("ALL AGGREGATION SELF-TESTS PASSED")
