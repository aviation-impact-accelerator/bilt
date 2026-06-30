__all__ = [
    "AnalysedTree",
    "IsolatedTree",
    "IsolationPlan",
    "OccurrenceProfile",
    "PlannedTree",
    "SimplifiedTree",
]

import typing
from dataclasses import dataclass

from bilt.symbolic._tree import Tree

if typing.TYPE_CHECKING:
    from bilt.symbolic._atoms import Atom
    from bilt.symbolic._expressions import Expr


@dataclass(frozen=True, slots=True)
class SimplifiedTree:
    tree: Tree


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class AnalysedTree:
    simplified_tree: SimplifiedTree
    target: Expr
    profile: OccurrenceProfile


@dataclass(frozen=True, slots=True)
class IsolationPlan:
    """Whole-tree normalisation before inverse-rule isolation.

    The only planning decision currently needed is whether to swap the relation
    sides before applying the inverse-rule isolator.

    Being already isolated is handled by the isolator's base case. Swapping
    sides is a single normalisation action, so it is represented as a boolean
    rather than as a strategy enum.
    """

    swap_sides: bool


@dataclass(frozen=True, slots=True)
class PlannedTree:
    simplified_tree: SimplifiedTree
    target: Expr
    profile: OccurrenceProfile
    plan: IsolationPlan


@dataclass(frozen=True, slots=True)
class IsolatedTree:
    relation_head: Atom
    target: Expr
    rhs: Expr

    def as_tree(self) -> Tree:
        return Tree(self.relation_head, (self.target.rep, self.rhs.rep))
