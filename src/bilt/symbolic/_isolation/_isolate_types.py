import typing
from dataclasses import dataclass
from enum import Enum, auto

from bilt.symbolic._isolation._expr_ops import same_expr, tree_hash
from bilt.symbolic._tree import Tree

if typing.TYPE_CHECKING:
    from bilt.symbolic._atoms import Atom
    from bilt.symbolic._expressions import Expr

# =============================================================================
# Typestate equation wrappers
# =============================================================================


@dataclass(frozen=True, slots=True, eq=False)
class SimplifiedTree:
    tree: Tree


@dataclass(frozen=True, slots=True, eq=False)
class OccurrenceProfile:
    lhs_count: int
    rhs_count: int

    @property
    def total_count(self) -> int:
        return self.lhs_count + self.rhs_count

    @property
    def target_missing(self) -> bool:
        return self.total_count == 0

    @property
    def target_on_both_sides(self) -> bool:
        return self.lhs_count > 0 and self.rhs_count > 0


@dataclass(frozen=True, slots=True, eq=False)
class AnalysedTree:
    simplified_tree: SimplifiedTree
    target: Expr
    profile: OccurrenceProfile


class IsolationStrategy(Enum):
    ALREADY_ISOLATED = auto()
    SWAP_SIDES = auto()
    INVERSE_RULES = auto()
    SWAP_THEN_INVERSE_RULES = auto()


@dataclass(frozen=True, slots=True, eq=False)
class IsolationPlan:
    strategy: IsolationStrategy


@dataclass(frozen=True, slots=True, eq=False)
class PlannedTree:
    simplified_tree: SimplifiedTree
    target: Expr
    profile: OccurrenceProfile
    plan: IsolationPlan


@dataclass(frozen=True, slots=True, eq=False)
class IsolatedTree:
    """Internal representation of an isolated binary relation tree.

    The original relation head is preserved, so isolating from an equality tree
    returns an equality tree, isolating from a less-than tree returns a less-than
    tree, and so on. This module does not currently reverse inequality direction;
    it only swaps sides when the target starts on the RHS.
    """

    relation_head: Atom
    target: Expr
    rhs: Expr

    @property
    def lhs(self) -> Expr:
        return self.target

    def as_tree(self) -> Tree:
        return Tree(self.relation_head, (self.target.rep, self.rhs.rep))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IsolatedTree):
            return NotImplemented

        return (
            self.relation_head == other.relation_head
            and same_expr(self.target, other.target)
            and same_expr(self.rhs, other.rhs)
        )

    def __hash__(self) -> int:
        return hash((self.relation_head, tree_hash(self.target.rep), tree_hash(self.rhs.rep)))
