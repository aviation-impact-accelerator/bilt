import typing

from bilt.symbolic._atoms import (
    ADD,
    INV,
    MUL,
    NEG,
    POW,
    FloatType,
    IntType,
    SymbolType,
)
from bilt.symbolic._errors import BiltSymbolicConversionError, BiltUnknownPortError
from bilt.symbolic._tree import Tree, float_tree, int_tree, relation_tree, symbol_tree

if typing.TYPE_CHECKING:
    import collections.abc

__all__ = [
    "Expr",
    "SubSystemExpr",
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
    """

    __slots__ = ("_rep",)

    _rep: Tree

    def __init__(self, rep: Tree) -> None:
        self._rep = rep

    @staticmethod
    def symbol(name: str) -> Expr:
        return symbol_expr(name)

    @staticmethod
    def integer(value: int) -> Expr:
        return Expr(int_tree(value))

    @staticmethod
    def real(value: float) -> Expr:
        return Expr(float_tree(value))

    @property
    def rep(self) -> Tree:
        return self._rep

    @property
    def tree(self) -> Tree:
        return self._rep

    @property
    def is_symbol(self) -> bool:
        return self._rep.is_leaf and self._rep.head.atom_type is SymbolType

    @property
    def is_number(self) -> bool:
        return self._rep.is_leaf and self._rep.head.atom_type in (IntType, FloatType)

    def __add__(self, other: object) -> Expr:
        return Expr(Tree(ADD, (self._rep, as_tree(other))))

    def __radd__(self, other: object) -> Expr:
        return Expr(Tree(ADD, (as_tree(other), self._rep)))

    def __sub__(self, other: object) -> Expr:
        return Expr(Tree(ADD, (self._rep, Tree(NEG, (as_tree(other),)))))

    def __rsub__(self, other: object) -> Expr:
        return Expr(Tree(ADD, (as_tree(other), Tree(NEG, (self._rep,)))))

    def __mul__(self, other: object) -> Expr:
        return Expr(Tree(MUL, (self._rep, as_tree(other))))

    def __rmul__(self, other: object) -> Expr:
        return Expr(Tree(MUL, (as_tree(other), self._rep)))

    def __truediv__(self, other: object) -> Expr:
        return Expr(Tree(MUL, (self._rep, Tree(INV, (as_tree(other),)))))

    def __rtruediv__(self, other: object) -> Expr:
        return Expr(Tree(MUL, (as_tree(other), Tree(INV, (self._rep,)))))

    def __pow__(self, other: object) -> Expr:
        return Expr(Tree(POW, (self._rep, as_tree(other))))

    def __rpow__(self, other: object) -> Expr:
        return Expr(Tree(POW, (as_tree(other), self._rep)))

    def __neg__(self) -> Expr:
        return Expr(Tree(NEG, (self._rep,)))

    def __pos__(self) -> Expr:
        return self

    # Expr deliberately overloads Python equality to construct symbolic relation
    # trees. This violates the normal object.__eq__ contract, which returns bool,
    # so static checkers need targeted suppressions here.
    def __eq__(self, other: object, /) -> Tree:  # pyrefly: ignore[bad-override]  # ty: ignore[invalid-method-override]
        return relation_tree("==", self._rep, as_tree(other))

    # See __eq__.
    def __ne__(self, other: object, /) -> Tree:  # pyrefly: ignore[bad-override]  # ty: ignore[invalid-method-override]
        return relation_tree("!=", self._rep, as_tree(other))

    def __lt__(self, other: object) -> Tree:
        return relation_tree("<", self._rep, as_tree(other))

    def __le__(self, other: object) -> Tree:
        return relation_tree("<=", self._rep, as_tree(other))

    def __gt__(self, other: object) -> Tree:
        return relation_tree(">", self._rep, as_tree(other))

    def __ge__(self, other: object) -> Tree:
        return relation_tree(">=", self._rep, as_tree(other))

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


class SubSystemExpr:
    """The sub-system an equation parameter stands for, whose ports are its symbols."""

    __slots__ = ("__name", "__ports")

    __name: str
    __ports: collections.abc.Mapping[str, str]

    def __init__(self, name: str, ports: collections.abc.Mapping[str, str]) -> None:
        """Build the sub-system known as name."""
        self.__name = name
        self.__ports = ports

    def __getattr__(self, port: str) -> Expr:
        """Return the leaf symbol standing for one of this sub-system's ports."""
        if port.startswith("__"):
            msg = f"{type(self).__name__!r} object has no attribute {port!r}"
            raise AttributeError(msg)

        symbol_name = self.__ports.get(port)
        if symbol_name is None:
            msg = f"{self.__name!r} has no port {port!r}. Valid ports: {', '.join(self.__ports)}"
            raise BiltUnknownPortError(msg)

        return symbol_expr(symbol_name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__name!r}, ports={tuple(self.__ports)!r})"


def as_tree(value: object) -> Tree:
    """Convert supported values to a symbolic Tree."""
    if isinstance(value, Expr):
        return value.rep

    if isinstance(value, Tree):
        return value

    if isinstance(value, bool):
        msg = "Boolean values cannot be converted to symbolic expressions."
        raise BiltSymbolicConversionError(msg)

    if isinstance(value, int):
        return int_tree(value)

    if isinstance(value, float):
        return float_tree(value)

    msg = f"Cannot convert {type(value).__name__!r} to symbolic Tree."
    raise BiltSymbolicConversionError(msg)


def as_expr(value: object) -> Expr:
    """Convert supported values to Expr."""
    if isinstance(value, Expr):
        return value

    return Expr(as_tree(value))


def symbol_expr(name: str) -> Expr:
    """Construct a symbolic expression placeholder."""
    return Expr(symbol_tree(name))
