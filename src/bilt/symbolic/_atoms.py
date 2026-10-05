import typing
import weakref

from bilt.symbolic._errors import BiltAtomMutationError

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.symbolic._dimensions import Coordinate, Dim

__all__ = [
    "ADD",
    "EQUALS",
    "GREATER_THAN",
    "GREATER_THAN_OR_EQUALS",
    "INV",
    "ISEL",
    "LESS_THAN",
    "LESS_THAN_OR_EQUALS",
    "MAX",
    "MEAN",
    "MIN",
    "MUL",
    "NEG",
    "NOT_EQUALS",
    "POW",
    "PROD",
    "SEL",
    "SUM",
    "Atom",
    "AtomType",
    "AxisType",
    "CoordinateType",
    "DimensionType",
    "FloatType",
    "IndexOpType",
    "IntType",
    "OpType",
    "ReductionOpType",
    "RelationOpType",
    "SymbolType",
    "atom_cache_size",
    "axis",
    "clear_atom_cache",
    "coordinate",
    "dimension",
    "floating",
    "integer",
    "relation_operator",
    "symbol",
]


_atom_cache: weakref.WeakValueDictionary[
    tuple[AtomType, collections.abc.Hashable],
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
ReductionOpType = AtomType("ReductionOp")
IndexOpType = AtomType("IndexOp")
CoordinateType = AtomType("Coordinate")
DimensionType = AtomType("Dimension")
AxisType = AtomType("Axis")
IntType = AtomType("Integer")
FloatType = AtomType("Float")


@typing.final
class Atom:
    """A hash-consed immutable leaf value.

    Every Atom has an AtomType and a value.

    Equality is identity-based. Symbolic equality is represented by the
    EQUALS atom inside a Tree, not by Atom.__eq__.
    """

    __slots__ = ("__weakref__", "_initialised", "atom_type", "value")

    atom_type: AtomType
    value: collections.abc.Hashable
    _initialised: bool

    def __new__(cls, atom_type: AtomType, value: collections.abc.Hashable) -> Atom:
        key = (atom_type, value)

        if (existing := _atom_cache.get(key)) is not None:
            return existing

        obj = object.__new__(cls)
        object.__setattr__(obj, "atom_type", atom_type)
        object.__setattr__(obj, "value", value)
        object.__setattr__(obj, "_initialised", True)

        return _atom_cache.setdefault(key, obj)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_initialised", False):
            msg = "Atom objects are immutable"
            raise BiltAtomMutationError(msg)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_initialised", False):
            msg = "Atom objects are immutable"
            raise BiltAtomMutationError(msg)
        object.__delattr__(self, name)

    def __repr__(self) -> str:
        return f"{self.atom_type}({self.value!r})"

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


def symbol(name: str) -> Atom:
    return Atom(SymbolType, name)


def dimension(value: Dim) -> Atom:
    return Atom(DimensionType, value)


def integer(value: int) -> Atom:
    return Atom(IntType, value)


def floating(value: float) -> Atom:
    return Atom(FloatType, value)


def axis(value: int) -> Atom:
    return Atom(AxisType, value)


def coordinate(value: Coordinate) -> Atom:
    return Atom(CoordinateType, value)


ADD = Atom(OpType, "add")
MUL = Atom(OpType, "mul")
POW = Atom(OpType, "pow")
NEG = Atom(OpType, "neg")
INV = Atom(OpType, "inv")

MEAN = Atom(ReductionOpType, "mean")
SUM = Atom(ReductionOpType, "sum")
MIN = Atom(ReductionOpType, "min")
MAX = Atom(ReductionOpType, "max")
PROD = Atom(ReductionOpType, "prod")

ISEL = Atom(IndexOpType, "isel")
SEL = Atom(IndexOpType, "sel")

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
