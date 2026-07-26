from .base import AssignmentContext, AssignmentPolicy, DeliveryFeedback, TierOption
from .greedy import AssignMaxPolicy
from .lyapunov import LyapunovPolicy
from .bandit import ContextualBanditPolicy

# Selected by --assignment_policy. All three share the AssignmentPolicy
# interface and live side by side permanently (TODO.md "Three-policy
# architecture"). greedy needs no lambda (E5's lambda=0 lemma); lyapunov and
# the bandit both require --policy_lambda (E3-gated) and raise
# NotImplementedError if constructed without one. All three are fully
# implemented as of 2026-07-24.
POLICIES = {
    "greedy": AssignMaxPolicy,
    "lyapunov": LyapunovPolicy,
    "bandit": ContextualBanditPolicy,
}

# Selected by --selection_policy. DIFFERENT decision surface from POLICIES
# above: which clients are in the round at all, not which tier a selected
# client trains (DIRECTION §3.2b). The two entries are Stage 1 leverage probes,
# not deployable policies -- see selection.py's docstring on why they are NOT
# ceilings.
from .selection import (  # noqa: E402
    SELECTION_POLICIES,
    CoverageSelector,
    OracleBandwidthSelector,
    SelectionContext,
    SelectionPolicy,
)

__all__ = [
    "AssignmentContext",
    "AssignmentPolicy",
    "DeliveryFeedback",
    "TierOption",
    "AssignMaxPolicy",
    "LyapunovPolicy",
    "ContextualBanditPolicy",
    "POLICIES",
    "SelectionPolicy",
    "SelectionContext",
    "OracleBandwidthSelector",
    "CoverageSelector",
    "SELECTION_POLICIES",
]
