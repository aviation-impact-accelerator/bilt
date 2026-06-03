from __future__ import annotations

import typing

from bilt.symbolic._atoms import FloatType, IntType, SymbolType
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

    # -- structural helpers -------------------------------------------------

    @property
    def is_symbol(self) -> bool:
        return self._rep.is_leaf and self._rep.head.atom_type is SymbolType

    @property
    def is_number(self) -> bool:
        return self._rep.is_leaf and self._rep.head.atom_type in (IntType, FloatType)

    @property
    def rep(self) -> Tree:
        """Underlying hash-consed tree representation."""
        return self._rep

    # -- equality -----------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Expr):
            return self._rep is other._rep
        return NotImplemented

    def __hash__(self) -> int:
        return id(self._rep)

    # -- display ------------------------------------------------------------

    def __repr__(self) -> str:
        return repr(self._rep)

    def __str__(self) -> str:
        return str(self._rep)
