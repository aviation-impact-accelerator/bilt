import typing

from bilt.symbolic._atoms import ADD, INV, MUL, NEG, POW, FloatType, IntType, SymbolType
from bilt.symbolic._relation_types import Relation
from bilt.symbolic._tree import Tree, float_tree, int_tree, symbol_tree

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

    # -- factories ----------------------------------------------------------

    @staticmethod
    def symbol(name: str) -> Expr:
        return Expr(symbol_tree(name))

    @staticmethod
    def integer(value: int) -> Expr:
        return Expr(int_tree(value))

    @staticmethod
    def real(value: float) -> Expr:
        return Expr(float_tree(value))

        # -- operator overloads -------------------------------------------------

    # Internally: no Sub or Div ops.
    #   a - b  -> add(a, neg(b))
    #   a / b  -> mul(a, inv(b))

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

    # -- structural helpers -------------------------------------------------

    @property
    def is_symbol(self) -> bool:
        return self._rep.is_leaf and self._rep.head.atom_type is SymbolType

    @property
    def is_number(self) -> bool:
        return self._rep.is_leaf and self._rep.head.atom_type in (IntType, FloatType)

    @property
    def rep(self) -> Tree:
        return self._rep

    # -- equality -----------------------------------------------------------

    def __eq__(self, other: object) -> Relation:  # ty: ignore[invalid-method-override]
        if isinstance(other, Expr):
            rhs = other
        elif isinstance(other, bool):
            return NotImplemented
        elif isinstance(other, int):
            rhs = Expr.integer(other)
        elif isinstance(other, float):
            rhs = Expr.real(other)
        else:
            return NotImplemented

        return Relation(lhs=self, rhs=rhs)

    def __hash__(self) -> int:
        return id(self._rep)

    # -- display ------------------------------------------------------------

    def __repr__(self) -> str:
        return repr(self._rep)

    def __str__(self) -> str:
        return str(self._rep)
