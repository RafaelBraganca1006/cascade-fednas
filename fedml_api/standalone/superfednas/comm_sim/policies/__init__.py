from .base import AssignmentContext, AssignmentPolicy, TierOption
from .greedy import AssignMaxPolicy
from .lyapunov import LyapunovPolicy
from .bandit import ContextualBanditPolicy

# Selected by --assignment_policy. All three share the AssignmentPolicy
# interface and live side by side permanently (TODO.md "Three-policy
# architecture") -- greedy is functional now (E5's lambda=0 lemma, no E3
# dependency); lyapunov/bandit are structural stubs until E3 sets lambda
# and the reward/context wiring lands (raise NotImplementedError if called).
POLICIES = {
    "greedy": AssignMaxPolicy,
    "lyapunov": LyapunovPolicy,
    "bandit": ContextualBanditPolicy,
}

__all__ = [
    "AssignmentContext",
    "AssignmentPolicy",
    "TierOption",
    "AssignMaxPolicy",
    "LyapunovPolicy",
    "ContextualBanditPolicy",
    "POLICIES",
]
