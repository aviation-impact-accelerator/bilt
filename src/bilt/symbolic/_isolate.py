import typing
from dataclasses import dataclass
from enum import Enum, auto

from bilt.symbolic._atoms import ADD, INV, MUL, NEG
from bilt.symbolic._expressions import Expr
from bilt.symbolic._tree import Tree
from bilt.symbolic._walk import contains, count_occurrences

if typing.TYPE_CHECKING:
    from bilt.symbolic._atoms import Atom

__all__ = [
    "IsolatedTree",
    "IsolationError",
    "TargetNotFoundError",
    "TargetOnBothSidesError",
    "UnsupportedIsolationError",
    "isolate",
]


BINARY_RELATION_ARG_COUNT = 2
UNARY_ARG_COUNT = 1


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


# =============================================================================
# Public API
# =============================================================================


def isolate(equation: Tree, target: Expr) -> Tree:
    """Return a relation tree with ``target`` isolated on the left-hand side.

    ``equation`` is expected to be a binary relation tree whose two children are
    expression trees. The returned tree keeps the same relation head as the input
    and has ``target.rep`` as its first child.
    """
    simplified = simplify_tree(equation)
    analysed = analyse_occurrences(simplified, target)
    planned = choose_isolation_plan(analysed)
    isolated = isolate_by_inverse_rules(planned)

    return IsolatedTree(
        relation_head=isolated.relation_head,
        target=isolated.target,
        rhs=simplify(isolated.rhs),
    ).as_tree()


# =============================================================================
# Pipeline stages
# =============================================================================


def simplify_tree(tree: Tree) -> SimplifiedTree:
    assert_binary_relation_tree(tree)

    lhs = simplify(tree_lhs_expr(tree))
    rhs = simplify(tree_rhs_expr(tree))

    return SimplifiedTree(make_relation_tree(tree.head, lhs, rhs))


def analyse_occurrences(
    simplified: SimplifiedTree,
    target: Expr,
) -> AnalysedTree:
    tree = simplified.tree
    lhs_count = count_tree_occurrences(tree_lhs_tree(tree), target.rep)
    rhs_count = count_tree_occurrences(tree_rhs_tree(tree), target.rep)

    profile = OccurrenceProfile(
        lhs_count=lhs_count,
        rhs_count=rhs_count,
    )

    if profile.target_missing:
        message = f"Target {target!r} does not occur in equation tree."
        raise TargetNotFoundError(message)

    if profile.target_on_both_sides:
        message = f"Target {target!r} occurs on both sides."
        raise TargetOnBothSidesError(message)

    if profile.total_count > 1:
        message = f"Target {target!r} occurs more than once."
        raise UnsupportedIsolationError(message)

    return AnalysedTree(
        simplified_tree=simplified,
        target=target,
        profile=profile,
    )


def choose_isolation_plan(analysed: AnalysedTree) -> PlannedTree:
    tree = analysed.simplified_tree.tree
    lhs = tree_lhs_expr(tree)
    rhs = tree_rhs_expr(tree)
    target = analysed.target

    if same_expr(lhs, target):
        strategy = IsolationStrategy.ALREADY_ISOLATED

    elif same_expr(rhs, target):
        strategy = IsolationStrategy.SWAP_SIDES

    elif expr_contains(lhs, target):
        strategy = IsolationStrategy.INVERSE_RULES

    elif expr_contains(rhs, target):
        strategy = IsolationStrategy.SWAP_THEN_INVERSE_RULES

    else:
        message = f"Target {target!r} does not occur in equation tree."
        raise TargetNotFoundError(message)

    return PlannedTree(
        simplified_tree=analysed.simplified_tree,
        target=target,
        profile=analysed.profile,
        plan=IsolationPlan(strategy),
    )


def isolate_by_inverse_rules(planned: PlannedTree) -> IsolatedTree:
    tree = planned.simplified_tree.tree
    relation_head = tree.head
    lhs = tree_lhs_expr(tree)
    rhs = tree_rhs_expr(tree)
    target = planned.target
    strategy = planned.plan.strategy

    if strategy is IsolationStrategy.ALREADY_ISOLATED:
        return IsolatedTree(relation_head=relation_head, target=target, rhs=rhs)

    if strategy is IsolationStrategy.SWAP_SIDES:
        return IsolatedTree(relation_head=relation_head, target=target, rhs=lhs)

    if strategy is IsolationStrategy.SWAP_THEN_INVERSE_RULES:
        lhs, rhs = rhs, lhs

    return isolate_expr_by_inverse_rules(
        lhs=lhs,
        rhs=rhs,
        target=target,
        relation_head=relation_head,
    )


# =============================================================================
# Inverse-rule isolation
# =============================================================================


def isolate_expr_by_inverse_rules(
    lhs: Expr,
    rhs: Expr,
    target: Expr,
    relation_head: Atom,
) -> IsolatedTree:
    lhs = simplify(lhs)
    rhs = simplify(rhs)

    if same_expr(lhs, target):
        return IsolatedTree(relation_head=relation_head, target=target, rhs=rhs)

    if is_neg(lhs):
        child = unary_child(lhs)
        return isolate_expr_by_inverse_rules(
            lhs=child,
            rhs=make_neg(rhs),
            target=target,
            relation_head=relation_head,
        )

    if is_inv(lhs):
        child = unary_child(lhs)
        return isolate_expr_by_inverse_rules(
            lhs=child,
            rhs=make_inv(rhs),
            target=target,
            relation_head=relation_head,
        )

    if is_add(lhs):
        new_lhs, new_rhs = isolate_add_expr(lhs, rhs, target)
        return isolate_expr_by_inverse_rules(
            lhs=new_lhs,
            rhs=new_rhs,
            target=target,
            relation_head=relation_head,
        )

    if is_mul(lhs):
        new_lhs, new_rhs = isolate_mul_expr(lhs, rhs, target)
        return isolate_expr_by_inverse_rules(
            lhs=new_lhs,
            rhs=new_rhs,
            target=target,
            relation_head=relation_head,
        )

    message = f"Cannot isolate {target!r} from {lhs!r}."
    raise UnsupportedIsolationError(message)


def isolate_add_expr(
    lhs: Expr,
    rhs: Expr,
    target: Expr,
) -> tuple[Expr, Expr]:
    terms = as_add_terms(lhs)
    target_terms, other_terms = split_exprs_by_target(terms, target)

    if len(target_terms) != 1:
        message = f"Expected one target-containing additive term, got {len(target_terms)}."
        raise UnsupportedIsolationError(message)

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
    target_factors, other_factors = split_exprs_by_target(factors, target)

    if len(target_factors) != 1:
        message = f"Expected one target-containing multiplicative factor, got {len(target_factors)}."
        raise UnsupportedIsolationError(message)

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
    child = simplify(unary_child(expr))

    if is_neg(child):
        return unary_child(child)

    return make_neg(child)


def simplify_inv(expr: Expr) -> Expr:
    child = simplify(unary_child(expr))

    if is_inv(child):
        return unary_child(child)

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


def split_exprs_by_target(
    exprs: tuple[Expr, ...],
    target: Expr,
) -> tuple[tuple[Expr, ...], tuple[Expr, ...]]:
    target_exprs: list[Expr] = []
    other_exprs: list[Expr] = []

    for expr in exprs:
        if expr_contains(expr, target):
            target_exprs.append(expr)
        else:
            other_exprs.append(expr)

    return tuple(target_exprs), tuple(other_exprs)


# =============================================================================
# Constructors
# =============================================================================


def make_relation_tree(relation_head: Atom, lhs: Expr, rhs: Expr) -> Tree:
    return Tree(relation_head, (lhs.rep, rhs.rep))


def make_add(*terms: Expr) -> Expr:
    if not terms:
        message = "Cannot build an empty addition."
        raise ValueError(message)

    if len(terms) == 1:
        return terms[0]

    return Expr(Tree(ADD, tuple(term.rep for term in terms)))


def make_mul(*factors: Expr) -> Expr:
    if not factors:
        message = "Cannot build an empty multiplication."
        raise ValueError(message)

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


def assert_binary_relation_tree(tree: Tree) -> None:
    if len(tree.args) != BINARY_RELATION_ARG_COUNT:
        message = (
            f"Expected a binary equation/relation tree with {BINARY_RELATION_ARG_COUNT} args, got {len(tree.args)}."
        )
        raise UnsupportedIsolationError(message)


def tree_lhs_tree(tree: Tree) -> Tree:
    assert_binary_relation_tree(tree)
    return tree.args[0]


def tree_rhs_tree(tree: Tree) -> Tree:
    assert_binary_relation_tree(tree)
    return tree.args[1]


def tree_lhs_expr(tree: Tree) -> Expr:
    return Expr(tree_lhs_tree(tree))


def tree_rhs_expr(tree: Tree) -> Expr:
    return Expr(tree_rhs_tree(tree))


def tree_args(tree: Tree) -> tuple[Tree, ...]:
    return tree.args


def expr_args(expr: Expr) -> tuple[Expr, ...]:
    return tuple(Expr(child) for child in tree_args(expr.rep))


def unary_child(expr: Expr) -> Expr:
    args = expr_args(expr)
    if len(args) != UNARY_ARG_COUNT:
        message = f"Expected unary expression {expr!r} to have {UNARY_ARG_COUNT} arg, got {len(args)}."
        raise UnsupportedIsolationError(message)

    return args[0]


def expr_contains(expr: Expr, target: Expr) -> bool:
    return contains(expr.rep, target.rep)


def count_tree_occurrences(tree: Tree, target: Tree) -> int:
    return count_occurrences(tree, target)


def tree_hash(tree: Tree) -> int:
    return hash((tree.head, tuple(tree_hash(child) for child in tree.args)))


def same_expr(left: Expr, right: Expr) -> bool:
    """Internal structural equality.

    Do not use Expr.__eq__, because Expr.__eq__ constructs a relation tree.
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
        same_tree(left_child, right_child) for left_child, right_child in zip(left.args, right.args, strict=True)
    )
