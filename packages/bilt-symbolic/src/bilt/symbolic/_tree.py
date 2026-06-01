import typing
import weakref

from bilt.symbolic._atoms import Atom, floating, integer, symbol
from bilt.symbolic._errors import TreeMutationError

_tree_cache: weakref.WeakValueDictionary[tuple[Atom, tuple[Tree, ...]], Tree] = weakref.WeakValueDictionary()

__all__ = [
    "Tree",
    "float_tree",
    "int_tree",
    "symbol_tree",
]


@typing.final
class Tree:
    __slots__ = ("__weakref__", "_locked", "args", "head")

    head: Atom
    args: tuple[Tree, ...]

    def __new__(cls, head: Atom, args: tuple[Tree, ...] = ()) -> Tree:
        key = (head, args)

        if (existing := _tree_cache.get(key)) is not None:
            return existing

        obj = object.__new__(cls)
        object.__setattr__(obj, "head", head)
        object.__setattr__(obj, "args", args)
        object.__setattr__(obj, "_locked", True)

        return _tree_cache.setdefault(key, obj)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            msg = f"{self.__class__.__name__} is immutable"
            raise TreeMutationError(msg)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_locked", False):
            msg = f"{self.__class__.__name__} is immutable"
            raise TreeMutationError(msg)
        object.__delattr__(self, name)

    @property
    def is_leaf(self) -> bool:
        """True when this node carries a value rather than an operation."""
        return not self.args

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"{self.__class__.__name__}({self.head!r})"
        args_repr = ", ".join(repr(a) for a in self.args)
        return f"{self.__class__.__name__}({self.head!r}, ({args_repr},))"

    def __str__(self) -> str:
        if self.is_leaf:
            return str(self.head)
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.head}({args_str})"


def symbol_tree(name: str) -> Tree:
    return Tree(symbol(name))


def int_tree(n: int) -> Tree:
    return Tree(integer(n))


def float_tree(x: float) -> Tree:
    return Tree(floating(x))
