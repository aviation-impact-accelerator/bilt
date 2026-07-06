from __future__ import annotations

import typing

from bilt.symbolic._atoms import (
    ADD,
    INV,
    MUL,
    NEG,
    POW,
    Atom,
    FloatType,
    IntType,
    SymbolType,
)
from bilt.symbolic._domains import Domain, DomainLike, as_domain, broadcast_domain
from bilt.symbolic._tree import Tree, float_tree, int_tree, relation_tree, symbol_tree

__all__ = [
    "Expr",
    "as_expr",
    "as_tree",
    "symbol_expr",
]


@typing.final
class Expr:
    """Thin user-facing symbolic expression wrapper.

    Expr exists to make Python syntax build canonical Tree objects.

    Tree remains the immutable/hash-consed storage layer.
    Expr is only the syntax layer used by @equation and other user-facing
    construction APIs.

    Domains describe the symbolic axes over which an expression is defined.
    """

    __slots__ = ("_rep",)

    _rep: Tree

    def __init__(self, rep: Tree) -> None:
        self._rep = rep

    @staticmethod
    def symbol(name: str, *, domain: DomainLike = None) -> Expr:
        return symbol_expr(name, domain=domain)

    @staticmethod
    def integer(value: int, *, domain: DomainLike = None) -> Expr:
        return Expr(int_tree(value, domain=_normalise_domain(domain)))

    @staticmethod
    def real(value: float, *, domain: DomainLike = None) -> Expr:
        return Expr(float_tree(value, domain=_normalise_domain(domain)))

    @property
    def rep(self) -> Tree:
        return self._rep

    @property
    def tree(self) -> Tree:
        return self._rep

    @property
    def domain(self) -> Domain:
        return self._rep.domain

    @property
    def is_scalar(self) -> bool:
        return self.domain.is_scalar

    @property
    def is_symbol(self) -> bool:
        return self._rep.is_leaf and self._rep.head.atom_type is SymbolType

    @property
    def is_number(self) -> bool:
        return self._rep.is_leaf and self._rep.head.atom_type in (IntType, FloatType)

    def __add__(self, other: object) -> Expr:
        rhs = as_tree(other)
        return _binary_expr(ADD, self._rep, rhs)

    def __radd__(self, other: object) -> Expr:
        lhs = as_tree(other)
        return _binary_expr(ADD, lhs, self._rep)

    def __sub__(self, other: object) -> Expr:
        rhs = as_tree(other)
        return _binary_expr(ADD, self._rep, _unary_tree(NEG, rhs))

    def __rsub__(self, other: object) -> Expr:
        lhs = as_tree(other)
        return _binary_expr(ADD, lhs, _unary_tree(NEG, self._rep))

    def __mul__(self, other: object) -> Expr:
        rhs = as_tree(other)
        return _binary_expr(MUL, self._rep, rhs)

    def __rmul__(self, other: object) -> Expr:
        lhs = as_tree(other)
        return _binary_expr(MUL, lhs, self._rep)

    def __truediv__(self, other: object) -> Expr:
        rhs = as_tree(other)
        return _binary_expr(MUL, self._rep, _unary_tree(INV, rhs))

    def __rtruediv__(self, other: object) -> Expr:
        lhs = as_tree(other)
        return _binary_expr(MUL, lhs, _unary_tree(INV, self._rep))

    def __pow__(self, other: object) -> Expr:
        rhs = as_tree(other)
        return _binary_expr(POW, self._rep, rhs)

    def __rpow__(self, other: object) -> Expr:
        lhs = as_tree(other)
        return _binary_expr(POW, lhs, self._rep)

    def __neg__(self) -> Expr:
        return Expr(_unary_tree(NEG, self._rep))

    def __pos__(self) -> Expr:
        return self

    # Expr deliberately overloads Python equality to construct symbolic relation
    # trees. This violates the normal object.__eq__ contract, which returns bool,
    # so static checkers need targeted suppressions here.
    def __eq__(self, other: object, /) -> Tree:  # pyrefly: ignore[bad-override]  # ty: ignore[invalid-method-override]
        rhs = as_tree(other)
        return _relation_tree("==", self._rep, rhs)

    # See __eq__.
    def __ne__(self, other: object, /) -> Tree:  # pyrefly: ignore[bad-override]  # ty: ignore[invalid-method-override]
        rhs = as_tree(other)
        return _relation_tree("!=", self._rep, rhs)

    def __lt__(self, other: object) -> Tree:
        rhs = as_tree(other)
        return _relation_tree("<", self._rep, rhs)

    def __le__(self, other: object) -> Tree:
        rhs = as_tree(other)
        return _relation_tree("<=", self._rep, rhs)

    def __gt__(self, other: object) -> Tree:
        rhs = as_tree(other)
        return _relation_tree(">", self._rep, rhs)

    def __ge__(self, other: object) -> Tree:
        rhs = as_tree(other)
        return _relation_tree(">=", self._rep, rhs)

    def __bool__(self) -> typing.NoReturn:
        msg = (
            "Symbolic expressions cannot be used as booleans. "
            "Return a symbolic relation from an @equation function instead."
        )
        raise TypeError(msg)

    def structurally_equal(self, other: object) -> bool:
        if not isinstance(other, Expr):
            return False

        return self.rep is other.rep

    def __hash__(self) -> int:
        return id(self._rep)

    def __repr__(self) -> str:
        return repr(self._rep)

    def __str__(self) -> str:
        return str(self._rep)


def as_tree(value: object, *, domain: DomainLike = None) -> Tree:
    """Convert supported values to a symbolic Tree.

    If ``domain`` is provided, only plain Python numbers are constructed with
    that domain. Existing Expr/Tree values keep their own domain.
    """
    if isinstance(value, Expr):
        if domain is not None and value.domain != as_domain(domain):
            msg = (
                "Cannot override the domain of an existing Expr in as_tree(). "
                "Create the expression with the required domain instead."
            )
            raise TypeError(msg)

        return value.rep

    if isinstance(value, Tree):
        if domain is not None and value.domain != as_domain(domain):
            msg = (
                "Cannot override the domain of an existing Tree in as_tree(). "
                "Create the tree with the required domain instead."
            )
            raise TypeError(msg)

        return value

    if isinstance(value, bool):
        msg = "Boolean values cannot be converted to symbolic expressions."
        raise TypeError(msg)

    normalised_domain = _normalise_domain(domain)

    if isinstance(value, int):
        return int_tree(value, domain=normalised_domain)

    if isinstance(value, float):
        return float_tree(value, domain=normalised_domain)

    msg = f"Cannot convert {type(value).__name__!r} to symbolic Tree."
    raise TypeError(msg)


def as_expr(value: object, *, domain: DomainLike = None) -> Expr:
    """Convert supported values to Expr."""
    if isinstance(value, Expr):
        if domain is not None and value.domain != as_domain(domain):
            msg = (
                "Cannot override the domain of an existing Expr in as_expr(). "
                "Create the expression with the required domain instead."
            )
            raise TypeError(msg)

        return value

    return Expr(as_tree(value, domain=domain))


def symbol_expr(name: str, *, domain: DomainLike = None) -> Expr:
    """Construct a symbolic expression placeholder."""
    return Expr(symbol_tree(name, domain=_normalise_domain(domain)))


def _normalise_domain(domain: DomainLike) -> Domain:
    if domain is None:
        return Domain.scalar()

    return as_domain(domain)


def _unary_tree(head: Atom, arg: Tree) -> Tree:
    return Tree(head, (arg,), domain=arg.domain)


def _binary_expr(head: Atom, lhs: Tree, rhs: Tree) -> Expr:
    return Expr(_binary_tree(head, lhs, rhs))


def _binary_tree(head: Atom, lhs: Tree, rhs: Tree) -> Tree:
    domain = broadcast_domain(lhs.domain, rhs.domain)
    return Tree(head, (lhs, rhs), domain=domain)


def _relation_tree(operator: str, lhs: Tree, rhs: Tree) -> Tree:
    domain = broadcast_domain(lhs.domain, rhs.domain)
    return relation_tree(operator, lhs, rhs, domain=domain)
