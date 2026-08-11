"""Aggregation weights under partial delivery -- DIRECTION.md 3.3c, "door two".

WHY THIS IS A MODULE AND NOT A LINE IN THE TRAINER
==================================================
The rule used to be two lines inside a 600-line training loop, which made it
impossible to test without a GPU and a six-hour run. Every silent bug this
project has paid for came from the aggregation path, and that path had no test
coverage. Extracting the rule as a pure function of OBSERVED state -- no torch,
no model, no side effects -- is what makes selftest_aggregation.py possible, and
the test is the point.

THE PROBLEM
===========
MaxNet weights from a fixed calendar: the client holding the round's "max slot"
gets beta(t), the rest split the remainder. That is well defined when every
update arrives whole. Anytime delivery breaks it -- the max-slot client may
deliver only a prefix, and the calendar does not say what that prefix is worth.

Three rules are implemented so they can be compared rather than argued about:

  SLOT          keep the pre-round slot weight whatever arrived (inherited).
  COMPLETENESS  scale it by delivered/intended bits (--maxnet_delivery_aware).
  CASCADE       multiply it by the inverse of how often this client reaches
                this region -- an unbiasedness correction, derived below.

Measured motivation: switching between the first two is worth +2.15 spread at
the canonical config (104232 vs 104125), larger than any policy effect measured
in this project. The aggregation axis has room.

THE DERIVATION
==============
Estimand: the update the server would have received if every selected client had
delivered in full.

Estimator: per coordinate, sum(w_i * delta_i) / sum(w_i) over the clients that
covered it -- which is what `_aggregate` already computes, and also what
HeteroFL, FjORD and FedRolex do (per-parameter coverage-count averaging). The
baseline is the field's standard, so any gain is a gain over the standard.

Correction: if client i reaches region s only a fraction pi(i, s) of the time,
the plain mean is biased toward the clients that arrive often, and weighting by
1/pi cancels that bias exactly. This is the standard inverse-probability result
-- FedAU (arXiv 2306.03401) Theorem 1 proves the optimal weight is 1/p -- applied
at the granularity this mechanism creates.

WHAT IS DIFFERENT HERE
======================
Prior corrections weight a CLIENT, because participation is binary for them:
either the update arrives or it does not. Anytime delivery makes participation
PARTIAL AND PER-REGION -- a client reaches the inner regions almost always and
the outer ones rarely -- so the quantity to invert is pi(i, s), per client and
per region. No prior method needs it because no prior method delivers partially.

Two facts make the granularity the right one rather than merely the novel one:

  - Under the block stream the drop rate is 0%, so the CLIENT marginal pi(i) is
    1 for everybody and carries no signal at all. A per-client correction would
    be inert here.
  - The REGION marginal is where the spread lives: measured shell coverage at
    the canonical config is 0.759 / 0.400 / 0.319 / 0.165, a 4.5x range.

Under correlated participation -- and mobile bandwidth is a sticky Markov chain,
so ours is correlated -- the object to invert is the stationary marginal rather
than a per-round probability, and omitting the correction leaves a bias term
that does not vanish with more rounds (arXiv 2410.01209).

WHY THE BASE STAYS MaxNet's SLOT WEIGHT
=======================================
The two factors answer different questions. The slot weight is a design
PREFERENCE about which region deserves emphasis; 1/pi is a CORRECTION for how
often that region was observed at all. Multiplying a preference by a correction
is standard practice, and it degenerates exactly: with complete delivery every
frequency is 1 and the rule returns the slot weights, so the base method is a
special case and any difference is attributable to the correction.

This also resolves the "third multiplicative weight" worry recorded in
DIRECTION 3.3: the failure that motivated it -- beta travelling with the slot --
came from stacking two PREFERENCES, which is a different error.

WHAT SELF-NORMALISATION DOES AND DOES NOT DO
============================================
`_aggregate` computes a per-coordinate weighted MEAN, so it is already
self-normalising: scaling every weight by a constant leaves the model unchanged.
Renormalising here therefore does NOT protect against a rescaling artefact -- an
earlier version of this file claimed it did, and that was wrong. Only RELATIVE
weights matter. It is kept because it is the Hajek estimator's defining step,
which trades a little bias for much lower variance when the frequencies are
estimated rather than known, and because it keeps logged weights comparable
across rules.
"""

DEFAULT_W_MAX = 4.0      # cap on the inverse-frequency correction
DEFAULT_LAPLACE = 1.0    # smoothing for the coverage-frequency estimate


class Delivery(object):
    """What the aggregation rule may look at for one client, all observed AFTER
    transmission.

    slot_weight: the weight MaxNet's calendar assigned to this client's slot.
        This is a design PREFERENCE about which region deserves emphasis and is
        kept as the base -- the correction below multiplies it rather than
        replacing it.
    delivered_tier: index of the largest tier whose shells all arrived, or -1
        when nothing complete arrived.
    num_tiers: size of the ladder.
    completeness: delivered bits / intended bits, for the `completeness` rule.
    coverage_freq: the estimate of pi(i, s) -- the long-run frequency with which
        THIS client's contribution reaches the region this delivery refreshed.
        None disables the correction, which makes the rule degenerate to the
        base weight.
    """

    def __init__(self, client_idx, slot_weight, delivered_tier, num_tiers,
                 completeness=1.0, coverage_freq=None):
        self.client_idx = client_idx
        self.slot_weight = float(slot_weight)
        self.delivered_tier = int(delivered_tier)
        self.num_tiers = int(num_tiers)
        self.completeness = float(completeness)
        self.coverage_freq = coverage_freq

    @property
    def delivered_max(self):
        return self.delivered_tier == self.num_tiers - 1

    @property
    def complete(self):
        return self.completeness >= 1.0 - 1e-9


def _renormalise(weights, target_mass):
    """Scale weights to sum to target_mass -- the Hajek (self-normalised) step.

    NOTE ON WHAT THIS DOES AND DOES NOT DO. `_aggregate` computes a per-
    coordinate WEIGHTED MEAN, sum(w_i * p_i) / sum(w_i), so it is already
    self-normalising: scaling every weight by a constant leaves the model
    unchanged. Renormalising here therefore does NOT protect the aggregate from
    a rescaling artefact -- an earlier version of this docstring claimed it did,
    and that was wrong. Only RELATIVE weights affect the result.

    It is kept for two real reasons: it is the Hajek estimator's defining step,
    which is what trades a little bias for much lower variance when the
    frequencies are estimated rather than known; and it keeps the logged weights
    on a comparable scale across rules, so a [aggregation] line means the same
    thing in every run.
    """
    total = sum(weights)
    if total <= 0:
        return list(weights)
    scale = target_mass / total
    return [w * scale for w in weights]


def slot_weights(deliveries):
    """SLOT: the inherited rule -- keep the pre-round slot weight."""
    return [d.slot_weight for d in deliveries]


def completeness_weights(deliveries):
    """COMPLETENESS: the shipped rule (--maxnet_delivery_aware).

    Scales the slot weight by the fraction of the intended payload that arrived.
    Deliberately not renormalised, which is the shipped behaviour: what it
    changes is the RELATIVE weight of truncated against complete deliveries, and
    that is the whole of its effect.
    """
    return [d.slot_weight * min(d.completeness, 1.0) for d in deliveries]


def inverse_frequency_correction(freq, w_max=DEFAULT_W_MAX):
    """The correction multiplier for a contribution to a region reached with
    estimated frequency `freq`.

        correction = min(1 / pi_hat, w_max)

    APPLIED PER REGION, NOT PER CLIENT, and that distinction is the whole point.
    A first implementation gave each client one scalar per round, indexed by the
    deepest region THAT client reached. It was inert by construction -- a client
    that habitually reaches region k reaches it almost always, so pi_hat was ~1
    for everybody -- and worse, when it did move it moved the wrong way: a
    well-connected client would have carried its correction onto the shared core
    too, increasing exactly the influence this correction exists to reduce.

    NO ROUND-LEVEL NORMALISATION IS NEEDED. `_aggregate` computes a per-
    coordinate weighted mean over the clients covering that coordinate, so the
    relative weights within a coordinate are all that reach the model. Applying
    the correction to the part that covers a region gives precisely the right
    relative weights among that region's contributors, and no global rescaling
    can improve on it.

    THE CAP IS WHERE UNBIASEDNESS IS TRADED FOR VARIANCE, and it is not a
    stability detail. Inverse-frequency weighting removes the bias only where
    every client reaches the region SOMETIMES (positivity). In the outermost
    region the poorly connected arrive on the order of 1% of rounds, so 1/pi
    reaches 50-100x: that is not correction, it is variance. The cap bounds it
    and, by bounding it, admits that residual bias remains there. See
    DIRECTION 3.3c.
    """
    if freq is None:
        return 1.0
    freq = min(max(float(freq), 1e-6), 1.0)
    return min(1.0 / freq, w_max)


def cascade_weights(deliveries, w_max=DEFAULT_W_MAX, laplace=DEFAULT_LAPLACE):
    """Per-client weights for the CASCADE rule.

    Kept for the paths that contribute a single, undifferentiated update (the
    shell path and all-or-nothing), where there is only one weight to give. The
    block stream uses inverse_frequency_correction per emitted part instead,
    because its parts map to regions and a single scalar cannot express a
    per-region correction.
    """
    if not deliveries:
        return []
    target_mass = sum(d.slot_weight for d in deliveries)
    weights = [
        d.slot_weight * inverse_frequency_correction(d.coverage_freq, w_max)
        for d in deliveries
    ]
    return _renormalise(weights, target_mass)


RULES = {
    "slot": slot_weights,
    "completeness": completeness_weights,
    "cascade": cascade_weights,
}
