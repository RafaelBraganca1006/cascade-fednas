"""Aggregation weights under partial delivery -- DIRECTION.md 3.3, "door two".

WHY THIS IS A MODULE AND NOT A LINE IN THE TRAINER
==================================================
The rule this file implements used to be two lines inside a 600-line training
loop, which made it impossible to test without a GPU and a six-hour run. Every
silent bug this project has paid for came from the aggregation path, and that
path has no test coverage at all. Extracting the rule as a pure function of
OBSERVED state -- no torch, no model, no side effects -- is what makes
selftest_aggregation.py possible, and the test is the point.

THE PROBLEM THE RULE SOLVES
===========================
MaxNet assigns a weight from a fixed calendar: the client holding the round's
"max slot" gets beta(t), everyone else splits the remainder. That is well
defined when every update arrives whole, which is true without truncation.
Anytime delivery breaks the assumption: the max-slot client may deliver only a
small prefix, and the calendar does not say what weight that prefix deserves.

Three specifications are defensible and this module implements all three, so
they can be compared rather than argued about:

  SLOT          keep the pre-round slot weight whatever arrived. The inherited
                behaviour. Vanilla SuperFedNAS never reaches this case.
  COMPLETENESS  scale the slot weight by delivered/intended bits
                (--maxnet_delivery_aware). Shipped, and the conservative choice
                because it yields the smaller spread.
  CASCADE       derive the weight from what ARRIVED rather than what was
                assigned, then modulate it by observed state: how starved the
                refreshed region is, and how rarely this client gets through.

Measured motivation for spending effort here: turning COMPLETENESS off is worth
+2.15 spread at the canonical config (104232 vs 104125), which is larger than
any policy effect measured in this project. The aggregation axis has room.

WHY CASCADE USES THE DELIVERED TIER
===================================
Two defects of COMPLETENESS, both recorded in DIRECTION 3.3 before this module
existed:

  1. It scales by bits_delivered / intended_bits, so the weight depends on what
     was ASSIGNED. Under blind sampling the assignment is random, which injects
     a random factor into an aggregation weight.
  2. Only clients that finish their plan keep full weight, and finishing
     requires bandwidth -- so it concentrates influence on the well-connected,
     the pathology this thesis argues against.

Deriving the weight from the delivered tier removes both: an arrived tier-k
update is weighted as a tier-k contributor, which is MaxNet's own semantics.

THE TWO OBSERVED-STATE FACTORS
==============================
  COVERAGE (DIRECTION 3.3 item 2). MaxNet decides from a schedule fixed before
  the run. Under dynamic bandwidth what actually arrived is known at aggregation
  time and is a strictly better signal. A delivery into a starved region is
  worth more than one into a region already refreshed by everybody.

  INVERSE DELIVERY RATE (DIRECTION 3.3 item 3, Horvitz-Thompson). A
  chronically poor client delivers nothing for many rounds and then gets one
  good window. Under non-IID its classes are otherwise absent from the model, so
  that single delivery is STANDING IN for the ones the channel destroyed, and
  1/p is exactly the size of the stand-in. This upgrades a fairness intuition
  into an unbiasedness argument.

THREE CONSTRAINTS, ALL REQUIRED OR THE RULE DESTABILISES
========================================================
  (a) HT variance explodes as p -> 0. Weights are therefore CLIPPED at w_max and
      the round is SELF-NORMALISED. Both are stated here rather than discovered
      as loss spikes.
  (b) p is the DELIVERY probability, not the selection probability. It is
      estimated from logged per-client delivery counts with Laplace smoothing,
      never assumed.
  (c) This would otherwise be the THIRD multiplicative weight source after
      MaxNet's beta and the completeness factor. Stacking weights is exactly
      what produced the "beta travels with the slot" case in the first place, so
      CASCADE REPLACES the calendar rather than multiplying into it.

MASS CONSERVATION IS WHAT MAKES THE COMPARISON VALID
====================================================
Every rule here returns weights summing to the same total as the slot rule for
that round. A rule that changed the total would rescale the aggregate, and the
resulting accuracy change would be an artefact of scaling rather than of
aggregation -- indistinguishable from a real effect in the results table, which
is precisely the class of silent error this module exists to prevent.
"""

DEFAULT_W_MAX = 4.0      # HT clip, in multiples of the mean slot weight
DEFAULT_LAPLACE = 1.0    # smoothing for the delivery-rate estimate


class Delivery(object):
    """What the aggregation rule may look at for one client, all observed AFTER
    transmission.

    slot_weight: the weight MaxNet's calendar assigned to this client's slot.
    delivered_tier: index of the largest tier whose shells all arrived, or -1
        when nothing complete arrived (the client may still have delivered loose
        blocks, which the collective rule handles).
    num_tiers: size of the ladder, so "is this the max tier" is decidable here
        rather than at the call site.
    completeness: delivered bits / intended bits, in [0, 1].
    region_coverage: observed coverage in [0, 1] of the region this delivery
        refreshed -- the fraction of that shell's units that have been arriving.
        None disables the coverage factor.
    delivery_count / round_count: how often this client has delivered at all,
        for the inverse-rate factor. None disables it.
    """

    def __init__(self, client_idx, slot_weight, delivered_tier, num_tiers,
                 completeness=1.0, region_coverage=None,
                 delivery_count=None, round_count=None):
        self.client_idx = client_idx
        self.slot_weight = float(slot_weight)
        self.delivered_tier = int(delivered_tier)
        self.num_tiers = int(num_tiers)
        self.completeness = float(completeness)
        self.region_coverage = region_coverage
        self.delivery_count = delivery_count
        self.round_count = round_count

    @property
    def delivered_max(self):
        """True when the update that ARRIVED is a full max-tier update."""
        return self.delivered_tier == self.num_tiers - 1

    @property
    def complete(self):
        return self.completeness >= 1.0 - 1e-9


def _renormalise(weights, target_mass):
    """Scale weights so they sum to target_mass.

    Self-normalisation, constraint (a). It is also what keeps every rule in this
    module comparable: two rules that disagree about relative influence but
    agree about total mass differ only in the thing under study.
    """
    total = sum(weights)
    if total <= 0:
        # Degenerate round (nobody delivered anything usable). Returning the
        # input unchanged is correct: the caller aggregates nothing, and
        # scaling zeros would divide by zero.
        return list(weights)
    scale = target_mass / total
    return [w * scale for w in weights]


def slot_weights(deliveries):
    """SLOT: the inherited rule -- keep the pre-round slot weight."""
    return [d.slot_weight for d in deliveries]


def completeness_weights(deliveries):
    """COMPLETENESS: the shipped rule (--maxnet_delivery_aware).

    Note it does NOT renormalise, which is the shipped behaviour and is why the
    round's total weight mass shrinks when deliveries are truncated. Preserved
    exactly so archived runs stay reproducible.
    """
    return [d.slot_weight * min(d.completeness, 1.0) for d in deliveries]


def cascade_weights(deliveries, w_max=DEFAULT_W_MAX, laplace=DEFAULT_LAPLACE,
                    use_coverage=True, use_inverse_rate=True):
    """CASCADE: derive the weight from what arrived, modulated by observed state.

    Steps, in order:

    1. BASE, from the DELIVERED tier rather than the assigned slot. The clients
       that actually delivered a max-tier update share the round's max-slot
       mass; everybody else shares the remainder. When several clients deliver
       the max tier the mass is SPLIT among them -- beta is a budget for the top
       of the ladder, not a per-client bonus, and applying it k times would
       inflate the round.

    2. COVERAGE. A delivery into a starved region counts for more. The factor is
       2 - coverage, so a never-refreshed region (coverage 0) counts double and a
       fully refreshed one (coverage 1) counts once. Bounded by construction,
       which matters more than tuning: an unbounded coverage factor is the same
       failure mode as unclipped HT.

    3. INVERSE DELIVERY RATE, Laplace-smoothed and clipped. p is estimated as
       (delivered + laplace) / (rounds + laplace), so a client that has never
       delivered gets a large but finite factor rather than a division by zero.

    4. CLIP at w_max multiples of the mean slot weight, then SELF-NORMALISE to
       the slot rule's total mass.

    Degeneracy, and it is a property the self-test asserts rather than a hope:
    when every client completes its plan and no observed-state factor is
    supplied, this returns the slot weights exactly. MaxNet is therefore a
    member of this rule's class, so a loss against it is attributable to the
    rule and not to a different action space -- the same discipline as
    `cap = XL` reproducing blind sampling.
    """
    if not deliveries:
        return []

    target_mass = sum(d.slot_weight for d in deliveries)
    n = len(deliveries)

    # ---- 1. base, from what ARRIVED --------------------------------------
    # The max-slot mass is whatever the calendar set aside for the top of the
    # ladder this round, i.e. the largest slot weight. Splitting it among the
    # clients that delivered the max tier keeps the round's top-of-ladder
    # influence constant regardless of how many got through.
    max_mass = max(d.slot_weight for d in deliveries)
    rest_mass = target_mass - max_mass

    winners = [d for d in deliveries if d.delivered_max]
    others = [d for d in deliveries if not d.delivered_max]

    base = {}
    if winners:
        for d in winners:
            base[d.client_idx] = max_mass / len(winners)
        for d in others:
            base[d.client_idx] = (rest_mass / len(others)) if others else 0.0
    else:
        # Nobody delivered the top tier -- the common case at a tight deadline,
        # where the complete max tier arrives under 2% of the time. The mass
        # earmarked for the top has nothing to land on, so the whole round is
        # shared evenly rather than handing a windfall to an arbitrary client.
        for d in deliveries:
            base[d.client_idx] = target_mass / n

    # ---- 2 & 3. observed-state factors, as a MULTIPLIER on the base -------
    # The clip bounds the AMPLIFICATION, not the final weight. That distinction
    # is not cosmetic: beta = 0.9 makes the max slot legitimately 7x the mean
    # weight by MaxNet's own design, so clipping the total would fight the
    # calendar instead of containing the estimator. What can actually explode is
    # 1/p as p -> 0, which is a property of the factor.
    #
    # Caught by test_degenerates_to_maxnet, which failed with 0.833 against
    # 0.900 when the clip was applied to the total: the max slot was cut to the
    # ceiling and renormalisation redistributed the difference. Complete
    # delivery must reproduce MaxNet exactly, or MaxNet is not a member of this
    # rule's class and a loss stops being attributable.
    weights = []
    for d in deliveries:
        factor = 1.0
        if use_coverage and d.region_coverage is not None:
            cov = min(max(float(d.region_coverage), 0.0), 1.0)
            factor *= (2.0 - cov)
        if (use_inverse_rate and d.delivery_count is not None
                and d.round_count is not None and d.round_count > 0):
            p = (float(d.delivery_count) + laplace) / (float(d.round_count) + laplace)
            p = min(max(p, 1e-6), 1.0)
            factor *= 1.0 / p
        factor = min(factor, w_max)
        weights.append(base[d.client_idx] * factor)

    # ---- 4. self-normalise to the slot rule's mass ------------------------
    # With no observed-state factors every factor is 1, the weights are the
    # base, and the base already sums to target_mass -- so this is the identity
    # and degeneracy is exact rather than approximate.
    return _renormalise(weights, target_mass)


RULES = {
    "slot": slot_weights,
    "completeness": completeness_weights,
    "cascade": cascade_weights,
}
