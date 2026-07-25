"""Contextual bandit assignment policy -- DIRECTION.md §3.2, the policy-chapter
contribution (E6). v1 design finalized 2026-07-24 after a risk pass (see
DIRECTION.md for the full writeup: one real gap found -- update() was never
called anywhere -- and four design hazards addressed before writing this).

Context (per assign() call, scores all tiers via one shared weight matrix --
NOT per-client models, too sample-starved at ~8 clients/round): debt per tier
normalized by (round_num+1) (keeps the staleness FRACTION meaningful
regardless of run length), log1p(bandwidth_bps/1e6) (log-Mbps), an explicit
has-bandwidth/has-loss indicator (both can be None), the client's last
training loss (default = ln(10), the uniform-prior cross-entropy for 10
classes, when not yet reported), bias term. Raw features span wildly
different magnitudes (debt ~1-100, bandwidth ~1e6-1e8, loss ~0.01-3) --
unnormalized, a linear model's dot product would be dominated entirely by
bandwidth, ignoring debt/loss regardless of true importance.

Reward (computed in update(), observed only for the CHOSEN arm -- standard
bandit setting): debt[delivered_tier] * bits_delivered / max_tier_payload,
discounted by (1 - lambda) if the delivery was cross-context. Mirrors
Lyapunov's objective (same units, normalized so raw bit counts ~1e8 don't
force an absurdly small learning rate) with one acknowledged simplification:
lambda discounts the WHOLE cross-context reward rather than isolating the
penalty to the S shell specifically, since DeliveryFeedback doesn't carry
shell-level bit granularity -- a deliberate v1 simplification, not an
oversight (Lyapunov keeps the more precise shell-by-shell accounting).

Update rule: NORMALIZED Widrow-Hoff / LMS online linear regression on the
chosen arm's weight vector only: w_k += (lr * error / ||phi||^2) * phi,
error = reward - w_k . phi. Normalizing by ||phi||^2 (not plain LMS) makes
pred_after = (1-lr)*pred_before + lr*reward exactly -- a convex combination
that provably never overshoots the target regardless of feature scale.
Caught by a self-test during development: plain (unnormalized) LMS with a
fixed lr overshot the target by >1000x under a large ||phi||^2 (an early-
round debt spike before the round_num+1 normalizer has shrunk it much).
Exploration:
epsilon-greedy with exponential decay, epsilon(round) = max(eps_min,
eps0 * decay^round) -- not a bandit-theory-optimal schedule (LinUCB's
confidence bounds would be more principled but add real numerical-stability
risk, matrix inversion of a growing covariance matrix, not worth it for v1).
Hyperparameters below are exploratory starting points, not calibrated --
same honesty as lambda=0.1 for Lyapunov's first run.

Pending-decision tracking: assign() stashes (features, chosen arm, max
payload, debt snapshot) keyed by (round, client_idx); update() pops and
consumes it. A missing key means this client's tier was NOT policy-decided
(the forced min/max coverage slots bypass assign() entirely, same as every
other TS-style sampler) -- update() silently no-ops rather than the trainer
needing to know which clients were policy-decided. This same path is also
the memory-leak guard: every assign() must be matched by exactly one
update() call, success OR total failure, checked by a self-test.
"""

import math
import random

from .base import AssignmentContext, AssignmentPolicy, DeliveryFeedback, TierOption

DEFAULT_LOSS = math.log(10)  # uniform-prior cross-entropy, 10 classes (CIFAR-10)


class ContextualBanditPolicy(AssignmentPolicy):
    def __init__(self, lam: float = None, lr: float = 0.05, eps0: float = 0.3,
                 eps_decay: float = 0.98, eps_min: float = 0.05, seed: int = 0):
        if lam is None:
            raise NotImplementedError(
                "ContextualBanditPolicy needs lambda from the E3 gradient-"
                "context ablation (DIRECTION §3.2), same reward units as "
                "LyapunovPolicy -- see module docstring."
            )
        self.lam = lam
        self.lr = lr
        self.eps0 = eps0
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self._rng = random.Random(seed)
        self._weights = None  # lazily sized (n_arms x feature_dim) on first assign()
        self._pending = {}  # (round, client_idx) -> (phi, arm_idx, max_payload, debt_snapshot)

    def _epsilon(self, round_num):
        return max(self.eps_min, self.eps0 * (self.eps_decay ** round_num))

    @staticmethod
    def _features(context: AssignmentContext, tiers):
        debt = context.debt_by_tier or {}
        denom = context.round_num + 1
        debt_feats = [debt.get(t.tier_id, 0.0) / denom for t in tiers]
        bw = context.bw_estimate_bps
        has_bw = 1.0 if bw is not None else 0.0
        bw_feat = math.log1p(bw / 1e6) if bw is not None else 0.0
        loss = context.data_utility.get("loss") if context.data_utility else None
        has_loss = 1.0 if loss is not None else 0.0
        loss_feat = loss if loss is not None else DEFAULT_LOSS
        return debt_feats + [bw_feat, has_bw, loss_feat, has_loss, 1.0]  # + bias

    @staticmethod
    def _dot(w, x):
        return sum(wi * xi for wi, xi in zip(w, x))

    def assign(self, context: AssignmentContext) -> TierOption:
        if not context.tier_options:
            raise ValueError("AssignmentContext.tier_options is empty")
        if context.debt_by_tier is None:
            raise ValueError(
                "ContextualBanditPolicy requires AssignmentContext.debt_by_tier "
                "-- wire the tier-debt table through policy_assign_sample."
            )
        tiers = sorted(context.tier_options, key=lambda t: t.payload_bits)
        phi = self._features(context, tiers)
        n_arms = len(tiers)
        if self._weights is None or len(self._weights) != n_arms:
            self._weights = [[0.0] * len(phi) for _ in range(n_arms)]

        if self._rng.random() < self._epsilon(context.round_num):
            arm_idx = self._rng.randrange(n_arms)
        else:
            scores = [self._dot(self._weights[i], phi) for i in range(n_arms)]
            best = max(scores)
            # tie-break toward the larger tier -- same convention as greedy/
            # Lyapunov; also gives a sensible cold-start bias (all-zero
            # weights -> all scores tie at 0 -> defaults to the max tier).
            arm_idx = max(i for i, s in enumerate(scores) if s == best)

        max_payload = tiers[-1].payload_bits
        self._pending[(context.round_num, context.client_idx)] = (
            phi, arm_idx, max_payload, dict(context.debt_by_tier),
        )
        return tiers[arm_idx]

    def update(self, feedback: DeliveryFeedback) -> None:
        pending = self._pending.pop((feedback.round_num, feedback.client_idx), None)
        if pending is None:
            return  # not policy-decided this round (e.g. forced min/max slot)
        phi, arm_idx, max_payload, debt_snapshot = pending
        bits = feedback.native_bits + feedback.cross_context_bits
        if max_payload <= 0 or bits <= 0 or feedback.delivered_tier is None:
            reward = 0.0  # total failure IS a signal: push the predicted score down
        else:
            raw = debt_snapshot.get(feedback.delivered_tier, 0.0) * bits / max_payload
            reward = raw if feedback.native_context else raw * (1.0 - self.lam)

        w = self._weights[arm_idx]
        error = reward - self._dot(w, phi)
        # Normalized LMS, not plain LMS: dividing the step by ||phi||^2 makes
        # pred_after = (1-lr)*pred_before + lr*reward exactly -- a convex
        # combination that PROVABLY never overshoots the target, regardless
        # of feature scale. Caught by a self-test: plain LMS with a fixed lr
        # overshot wildly (81041 vs a target of 50) under a large ||phi||^2
        # (e.g. an early-round debt spike before the round_num+1 normalizer
        # has had time to shrink it) -- this is the fix, not a test-only workaround.
        norm = sum(x * x for x in phi) + 1e-8
        step = self.lr * error / norm
        for i in range(len(w)):
            w[i] += step * phi[i]
