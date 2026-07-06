from __future__ import annotations

__all__ = [
    "ADD",
    "EQUALS",
    "GREATER_THAN",
    "GREATER_THAN_OR_EQUALS",
    "INV",
    "LESS_THAN",
    "LESS_THAN_OR_EQUALS",
    "MUL",
    "NEG",
    "NOT_EQUALS",
    "POW",
    "Atom",
    "AtomType",
    "FloatType",
    "IntType",
    "OpType",
    "RelationOpType",
    "SymbolType",
    "atom_cache_size",
    "clear_atom_cache",
    "floating",
    "integer",
    "relation_operator",
    "symbol",
    "symbol_domain",
]

import typing
import weakref

from bilt.symbolic._domains import Domain, DomainLike, as_domain
from bilt.symbolic._errors import AtomMutationError

if typing.TYPE_CHECKING:
    import collections.abc

_atom_cache: weakref.WeakValueDictionary[
    tuple[AtomType, collections.abc.Hashable, Domain],
    Atom,
] = weakref.WeakValueDictionary()


@typing.final
class AtomType:
    """Identity-based metadata tag that distinguishes kinds of atom.

    Two AtomType objects with the same name are still distinct atom types.

    Example:
        Symbol = AtomType("Symbol")
        Op = AtomType("Op")

        Symbol("x") is Symbol("x")
        Symbol("x") is not Op("x")
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __call__(self, value: collections.abc.Hashable) -> Atom:
        return Atom(self, value)


SymbolType = AtomType("Symbol")
OpType = AtomType("Op")
RelationOpType = AtomType("RelationOp")
IntType = AtomType("Integer")
FloatType = AtomType("Float")


@typing.final
class Atom:
    """A hash-consed immutable leaf value.

    Every Atom has an AtomType, a value, and optional symbolic domain metadata.

    Equality is identity-based. Symbolic equality is represented by the
    EQUALS atom inside a Tree, not by Atom.__eq__.
    """

    __slots__ = (
        "__weakref__",
        "_initialised",
        "atom_type",
        "domain",
        "value",
    )

    atom_type: AtomType
    value: collections.abc.Hashable
    domain: Domain
    _initialised: bool

    def __new__(
        cls,
        atom_type: AtomType,
        value: collections.abc.Hashable,
        *,
        domain: DomainLike = None,
    ) -> Atom:
        atom_domain = as_domain(domain)
        key = (atom_type, value, atom_domain)

        if (existing := _atom_cache.get(key)) is not None:
            return existing

        obj = object.__new__(cls)
        object.__setattr__(obj, "atom_type", atom_type)
        object.__setattr__(obj, "value", value)
        object.__setattr__(obj, "domain", atom_domain)
        object.__setattr__(obj, "_initialised", True)

        return _atom_cache.setdefault(key, obj)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_initialised", False):
            msg = "Atom objects are immutable"
            raise AtomMutationError(msg)

        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_initialised", False):
            msg = "Atom objects are immutable"
            raise AtomMutationError(msg)

        object.__delattr__(self, name)

    def __repr__(self) -> str:
        if self.atom_type is SymbolType and self.domain != Domain.scalar():
            return f"{self.atom_type}({self.value!r}, domain={self.domain!r})"

        return f"{self.atom_type}({self.value!r})"

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


def symbol(
    name: str,
    *,
    domain: DomainLike = None,
) -> Atom:
    return Atom(SymbolType, name, domain=domain)


def symbol_domain(atom: Atom) -> Domain:
    if atom.atom_type is not SymbolType:
        return Domain.scalar()

    return atom.domain


def integer(value: int) -> Atom:
    return Atom(IntType, value)


def floating(value: float) -> Atom:
    return Atom(FloatType, value)


ADD = Atom(OpType, "add")
MUL = Atom(OpType, "mul")
POW = Atom(OpType, "pow")
NEG = Atom(OpType, "neg")
INV = Atom(OpType, "inv")

EQUALS = Atom(RelationOpType, "==")
NOT_EQUALS = Atom(RelationOpType, "!=")
LESS_THAN = Atom(RelationOpType, "<")
LESS_THAN_OR_EQUALS = Atom(RelationOpType, "<=")
GREATER_THAN = Atom(RelationOpType, ">")
GREATER_THAN_OR_EQUALS = Atom(RelationOpType, ">=")


_RELATION_OPERATORS: dict[str, Atom] = {
    "==": EQUALS,
    "!=": NOT_EQUALS,
    "<": LESS_THAN,
    "<=": LESS_THAN_OR_EQUALS,
    ">": GREATER_THAN,
    ">=": GREATER_THAN_OR_EQUALS,
}


def relation_operator(value: str) -> Atom:
    """Return the built-in relation operator atom for a symbolic operator string."""
    try:
        return _RELATION_OPERATORS[value]
    except KeyError:
        msg = f"Unknown relation operator: {value!r}"
        raise ValueError(msg) from None


def atom_cache_size() -> int:
    """Return the number of currently live interned atoms.

    This is mainly useful for debugging and tests.
    """
    return len(_atom_cache)


def clear_atom_cache() -> None:
    """Clear the atom cache.

    This is mainly useful in tests. Normal code should not usually need this.
    """
    _atom_cache.clear()
