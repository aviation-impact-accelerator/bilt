from dataclasses import dataclass
from typing import TYPE_CHECKING

from bilt.symbolic._errors import IsolationError
from bilt.symbolic._expressions import ADD, Expr
from bilt.symbolic._walk import contains

if TYPE_CHECKING:
    from bilt.symbolic._tree import Tree

BINARY_LENGTH = 2


def _is_symbol(expr: Expr) -> bool:
    return expr.is_symbol


@dataclass(frozen=True, slots=True)
class SymbolTarget:
    expr: Expr

    @property
    def rep(self) -> Tree:
        return self.expr.rep

    @classmethod
    def from_expr(cls, expr: Expr) -> SymbolTarget:
        if not _is_symbol(expr):
            msg = "Expected a single symbolic variable."
            raise TypeError(msg)

        return cls(expr)


@dataclass(frozen=True, slots=True)
class BinaryArgs:
    left: Tree
    right: Tree

    @classmethod
    def from_args(cls, args: tuple[Tree, ...], op_name: str) -> BinaryArgs:
        if len(args) != BINARY_LENGTH:
            msg = f"Expected binary {op_name}, got {len(args)} arguments."
            raise IsolationError(msg)

        return cls(left=args[0], right=args[1])


def peel(lhs: Expr, rhs: Expr, target: SymbolTarget) -> Expr:
    """Given lhs == rhs, peel operations from lhs until target is isolated.

    Assumes target occurs exactly once inside lhs.
    """
    if lhs.rep is target.rep:
        return rhs

    rep = lhs.rep

    if rep.head is ADD:
        args = BinaryArgs.from_args(rep.args, "addition")
        return _peel_add(args, rhs, target)

    msg = f"Cannot isolate through unsupported expression head: {rep.head!r}"
    raise IsolationError(msg)


def _peel_add(args: BinaryArgs, rhs: Expr, target: SymbolTarget) -> Expr:
    left = Expr(args.left)
    right = Expr(args.right)

    left_has = contains(left.rep, target.rep)
    right_has = contains(right.rep, target.rep)

    if left_has and right_has:
        msg = "Target occurs in both operands of addition."
        raise IsolationError(msg)

    if left_has:
        return peel(left, rhs + (-right), target)

    if right_has:
        return peel(right, rhs + (-left), target)

    msg = "Target not found in addition."
    raise IsolationError(msg)
