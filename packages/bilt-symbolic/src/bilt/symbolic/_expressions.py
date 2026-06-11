import typing

from bilt.symbolic._atoms import ADD, INV, MUL, NEG, POW, FloatType, IntType, SymbolType
from bilt.symbolic._equalities import Equality
from bilt.symbolic._tree import Tree, float_tree, int_tree, symbol_tree

if typing.TYPE_CHECKING:
    import types

__all__ = [
    "Expr",
]


@typing.final
class Expr:
    """User-facing symbolic expression."""

    __slots__ = ("_rep",)

    _rep: Tree

    def __init__(self, rep: Tree) -> None:
        self._rep = rep

    @staticmethod
    def symbol(name: str) -> Expr:
        return Expr(symbol_tree(name))

    @staticmethod
    def integer(value: int) -> Expr:
        return Expr(int_tree(value))

    @staticmethod
    def real(value: float) -> Expr:
        return Expr(float_tree(value))

    def __add__(self, other: Expr) -> Expr:
        return Expr(Tree(ADD, (self._rep, other._rep)))

    def __mul__(self, other: Expr) -> Expr:
        return Expr(Tree(MUL, (self._rep, other._rep)))

    def __pow__(self, other: Expr) -> Expr:
        return Expr(Tree(POW, (self._rep, other._rep)))

    def __neg__(self) -> Expr:
        return Expr(Tree(NEG, (self._rep,)))

    def __sub__(self, other: Expr) -> Expr:
        return self + (-other)

    def __truediv__(self, other: Expr) -> Expr:
        return Expr(Tree(MUL, (self._rep, Tree(INV, (other._rep,)))))

    @property
    def is_symbol(self) -> bool:
        return self._rep.is_leaf and self._rep.head.atom_type is SymbolType

    @property
    def is_number(self) -> bool:
        return self._rep.is_leaf and self._rep.head.atom_type in (IntType, FloatType)

    @property
    def rep(self) -> Tree:
        return self._rep

    def __eq__(self, other: object) -> Equality:  # ty: ignore[invalid-method-override]
        rhs = _rhs_as_expr(other)
        if rhs is NotImplemented:
            return NotImplemented

        return Equality(lhs=self, rhs=rhs)

    def structurally_equal(self, other: object) -> bool:
        if not isinstance(other, Expr):
            return False

        return self.rep == other.rep

    def __hash__(self) -> int:
        return id(self._rep)

    def __repr__(self) -> str:
        return repr(self._rep)

    def __str__(self) -> str:
        return str(self._rep)


def _rhs_as_expr(value: object) -> Expr | types.NotImplementedType:
    if isinstance(value, Expr):
        return value

    if isinstance(value, bool):
        return NotImplemented

    if isinstance(value, int):
        return Expr.integer(value)

    if isinstance(value, float):
        return Expr.real(value)

    return NotImplemented
