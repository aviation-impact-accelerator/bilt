from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from bilt.symbolic._atoms import ADD, INV, MUL, NEG
from bilt.symbolic._expressions import Expr
from bilt.symbolic._relations import Relation
from bilt.symbolic._tree import Tree
from bilt.symbolic._walk import contains, count_occurrences


__all__ = [
    "isolate",
    "IsolatedRelation",
    "IsolationError",
    "TargetNotFoundError",
    "TargetOnBothSidesError",
    "UnsupportedIsolationError",
]


# =============================================================================
# Errors
# =============================================================================


class IsolationError(Exception):
    pass


class TargetNotFoundError(IsolationError):
    pass


class TargetOnBothSidesError(IsolationError):
    pass


class UnsupportedIsolationError(IsolationError):
    pass


# =============================================================================
# Typestate relation wrappers
# =============================================================================


@dataclass(frozen=True, slots=True, eq=False)
class RawRelation:
    relation: Relation


@dataclass(frozen=True, slots=True, eq=False)
class SimplifiedRelation:
    relation: Relation


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
class AnalysedRelation:
    relation: SimplifiedRelation
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
class PlannedRelation:
    relation: SimplifiedRelation
    target: Expr
    profile: OccurrenceProfile
    plan: IsolationPlan


@dataclass(frozen=True, slots=True, eq=False)
class IsolatedRelation:
    target: Expr
    rhs: Expr

    @property
    def lhs(self) -> Expr:
        return self.target

    def as_relation(self) -> Relation:
        return Relation(lhs=self.target, rhs=self.rhs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IsolatedRelation):
            return NotImplemented

        return same_expr(self.target, other.target) and same_expr(self.rhs, other.rhs)


# =============================================================================
# Public API
# =============================================================================


def isolate(relation: Relation, target: Expr) -> IsolatedRelation:

    simplified = simplify_relation(relation)
    analysed = analyse_occurrences(simplified, target)
    planned = choose_isolation_plan(analysed)

    isolated = isolate_by_inverse_rules(planned)

    return IsolatedRelation(
        target=isolated.target,
        rhs=simplify(isolated.rhs),
    )


# =============================================================================
# Pipeline stages
# =============================================================================


def simplify_relation(relation: Relation) -> SimplifiedRelation:

    simplified = Relation(
        lhs=simplify(relation.lhs),
        rhs=simplify(relation.rhs),
    )

    return SimplifiedRelation(simplified)


def analyse_occurrences(
    relation: SimplifiedRelation,
    target: Expr,
) -> AnalysedRelation:
    lhs_count = count_expr_occurrences(relation.relation.lhs, target)
    rhs_count = count_expr_occurrences(relation.relation.rhs, target)

    profile = OccurrenceProfile(
        lhs_count=lhs_count,
        rhs_count=rhs_count,
    )

    if profile.target_missing:
        raise TargetNotFoundError(f"Target {target!r} does not occur in relation.")

    if profile.target_on_both_sides:
        raise TargetOnBothSidesError(
            f"Target {target!r} occurs on both sides."
        )

    if profile.total_count > 1:
        raise UnsupportedIsolationError(
            f"Target {target!r} occurs more than once."
        )

    return AnalysedRelation(
        relation=relation,
        target=target,
        profile=profile,
    )


def choose_isolation_plan(analysed: AnalysedRelation) -> PlannedRelation:
    relation = analysed.relation.relation
    target = analysed.target

    if same_expr(relation.lhs, target):
        strategy = IsolationStrategy.ALREADY_ISOLATED

    elif same_expr(relation.rhs, target):
        strategy = IsolationStrategy.SWAP_SIDES

    elif expr_contains(relation.lhs, target):
        strategy = IsolationStrategy.INVERSE_RULES

    elif expr_contains(relation.rhs, target):
        strategy = IsolationStrategy.SWAP_THEN_INVERSE_RULES

    else:
        raise TargetNotFoundError(f"Target {target!r} does not occur in relation.")

    return PlannedRelation(
        relation=analysed.relation,
        target=target,
        profile=analysed.profile,
        plan=IsolationPlan(strategy),
    )


def isolate_by_inverse_rules(planned: PlannedRelation) -> IsolatedRelation:
    relation = planned.relation.relation
    target = planned.target
    strategy = planned.plan.strategy

    if strategy is IsolationStrategy.ALREADY_ISOLATED:
        return IsolatedRelation(target=target, rhs=relation.rhs)

    if strategy is IsolationStrategy.SWAP_SIDES:
        return IsolatedRelation(target=target, rhs=relation.lhs)

    if strategy is IsolationStrategy.SWAP_THEN_INVERSE_RULES:
        relation = Relation(lhs=relation.rhs, rhs=relation.lhs)

    return isolate_expr_by_inverse_rules(
        lhs=relation.lhs,
        rhs=relation.rhs,
        target=target,
    )


# =============================================================================
# Inverse-rule isolation
# =============================================================================


def isolate_expr_by_inverse_rules(
    lhs: Expr,
    rhs: Expr,
    target: Expr,
) -> IsolatedRelation:
    lhs = simplify(lhs)
    rhs = simplify(rhs)

    if same_expr(lhs, target):
        return IsolatedRelation(target=target, rhs=rhs)

    if is_neg(lhs):
        child = expr_args(lhs)[0]
        return isolate_expr_by_inverse_rules(
            lhs=child,
            rhs=make_neg(rhs),
            target=target,
        )

    if is_inv(lhs):
        child = expr_args(lhs)[0]
        return isolate_expr_by_inverse_rules(
            lhs=child,
            rhs=make_inv(rhs),
            target=target,
        )

    if is_add(lhs):
        new_lhs, new_rhs = isolate_add_expr(lhs, rhs, target)
        return isolate_expr_by_inverse_rules(new_lhs, new_rhs, target)

    if is_mul(lhs):
        new_lhs, new_rhs = isolate_mul_expr(lhs, rhs, target)
        return isolate_expr_by_inverse_rules(new_lhs, new_rhs, target)

    raise UnsupportedIsolationError(
        f"Cannot isolate {target!r} from {lhs!r}."
    )


def isolate_add_expr(
    lhs: Expr,
    rhs: Expr,
    target: Expr,
) -> tuple[Expr, Expr]:
    terms = as_add_terms(lhs)
    target_terms, other_terms = split_terms_by_target(terms, target)

    if len(target_terms) != 1:
        raise UnsupportedIsolationError(
            f"Expected one target-containing additive term, "
            f"got {len(target_terms)}."
        )

    target_part = target_terms[0]

    if not other_terms:
        return target_part, rhs

    other_part = make_add(*other_terms)
    new_rhs = make_sub(rhs, other_part)

    return simplify(target_part), simplify(new_rhs)


def isolate_mul_expr(
    lhs: Expr,
    rhs: Expr,
    target: Expr,
) -> tuple[Expr, Expr]:
    factors = as_mul_factors(lhs)
    target_factors, other_factors = split_factors_by_target(factors, target)

    if len(target_factors) != 1:
        raise UnsupportedIsolationError(
            f"Expected one target-containing multiplicative factor, "
            f"got {len(target_factors)}."
        )

    target_part = target_factors[0]

    if not other_factors:
        return target_part, rhs

    other_part = make_mul(*other_factors)
    new_rhs = make_div(rhs, other_part)

    return simplify(target_part), simplify(new_rhs)


# =============================================================================
# Simplification
# =============================================================================


def simplify(expr: Expr) -> Expr:
    if is_add(expr):
        return simplify_add(expr)

    if is_mul(expr):
        return simplify_mul(expr)

    if is_neg(expr):
        return simplify_neg(expr)

    if is_inv(expr):
        return simplify_inv(expr)

    return expr


def simplify_add(expr: Expr) -> Expr:
    terms = tuple(simplify(term) for term in as_add_terms(expr))

    flattened: list[Expr] = []
    for term in terms:
        if is_add(term):
            flattened.extend(as_add_terms(term))
        else:
            flattened.append(term)

    if len(flattened) == 1:
        return flattened[0]

    return make_add(*flattened)


def simplify_mul(expr: Expr) -> Expr:
    factors = tuple(simplify(factor) for factor in as_mul_factors(expr))

    flattened: list[Expr] = []
    for factor in factors:
        if is_mul(factor):
            flattened.extend(as_mul_factors(factor))
        else:
            flattened.append(factor)

    if len(flattened) == 1:
        return flattened[0]

    return make_mul(*flattened)


def simplify_neg(expr: Expr) -> Expr:
    child = simplify(expr_args(expr)[0])

    if is_neg(child):
        return expr_args(child)[0]

    return make_neg(child)


def simplify_inv(expr: Expr) -> Expr:
    child = simplify(expr_args(expr)[0])

    if is_inv(child):
        return expr_args(child)[0]

    return make_inv(child)


# =============================================================================
# Expression decomposition
# =============================================================================


def as_add_terms(expr: Expr) -> tuple[Expr, ...]:
    if is_add(expr):
        return expr_args(expr)

    return (expr,)


def as_mul_factors(expr: Expr) -> tuple[Expr, ...]:
    if is_mul(expr):
        return expr_args(expr)

    return (expr,)


def split_terms_by_target(
    terms: tuple[Expr, ...],
    target: Expr,
) -> tuple[tuple[Expr, ...], tuple[Expr, ...]]:
    target_terms: list[Expr] = []
    other_terms: list[Expr] = []

    for term in terms:
        if expr_contains(term, target):
            target_terms.append(term)
        else:
            other_terms.append(term)

    return tuple(target_terms), tuple(other_terms)


def split_factors_by_target(
    factors: tuple[Expr, ...],
    target: Expr,
) -> tuple[tuple[Expr, ...], tuple[Expr, ...]]:
    target_factors: list[Expr] = []
    other_factors: list[Expr] = []

    for factor in factors:
        if expr_contains(factor, target):
            target_factors.append(factor)
        else:
            other_factors.append(factor)

    return tuple(target_factors), tuple(other_factors)


# =============================================================================
# Constructors
# =============================================================================


def make_add(*terms: Expr) -> Expr:
    if not terms:
        raise ValueError("Cannot build an empty addition.")

    if len(terms) == 1:
        return terms[0]

    return Expr(Tree(ADD, tuple(term.rep for term in terms)))


def make_mul(*factors: Expr) -> Expr:
    if not factors:
        raise ValueError("Cannot build an empty multiplication.")

    if len(factors) == 1:
        return factors[0]

    return Expr(Tree(MUL, tuple(factor.rep for factor in factors)))


def make_neg(expr: Expr) -> Expr:
    return Expr(Tree(NEG, (expr.rep,)))


def make_inv(expr: Expr) -> Expr:
    return Expr(Tree(INV, (expr.rep,)))


def make_sub(lhs: Expr, rhs: Expr) -> Expr:
    return make_add(lhs, make_neg(rhs))


def make_div(lhs: Expr, rhs: Expr) -> Expr:
    return make_mul(lhs, make_inv(rhs))


# =============================================================================
# Predicates
# =============================================================================


def is_add(expr: Expr) -> bool:
    return expr.rep.head is ADD


def is_mul(expr: Expr) -> bool:
    return expr.rep.head is MUL


def is_neg(expr: Expr) -> bool:
    return expr.rep.head is NEG


def is_inv(expr: Expr) -> bool:
    return expr.rep.head is INV


# =============================================================================
# Tree / expression helpers
# =============================================================================


def tree_args(tree: Tree) -> tuple[Tree, ...]:
    return tree.args


def expr_args(expr: Expr) -> tuple[Expr, ...]:
    return tuple(Expr(child) for child in tree_args(expr.rep))


def expr_contains(expr: Expr, target: Expr) -> bool:
    return contains(expr.rep, target.rep)


def count_expr_occurrences(expr: Expr, target: Expr) -> int:
    return count_occurrences(expr.rep, target.rep)


def same_expr(left: Expr, right: Expr) -> bool:
    """
    Internal structural equality.

    Do not use Expr.__eq__, because Expr.__eq__ constructs a Relation.
    """
    return same_tree(left.rep, right.rep)


def same_tree(left: Tree, right: Tree) -> bool:
    if left is right:
        return True

    if left.head != right.head:
        return False

    if len(left.args) != len(right.args):
        return False

    return all(
        same_tree(left_child, right_child)
        for left_child, right_child in zip(left.args, right.args, strict=True)
    )