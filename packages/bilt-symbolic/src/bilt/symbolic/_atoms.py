import collections.abc
import typing
import weakref

from bilt.symbolic._errors import AtomMutationError

if typing.TYPE_CHECKING:
    import collections.abc

__all__ = [
    "Atom",
]


_atom_cache: weakref.WeakValueDictionary[
    tuple[object, collections.abc.Hashable],
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


@typing.final
class Atom:
    """A hash-consed immutable leaf value.

    Every Atom has an AtomType and a value. Equality is identity-based:
    hash-consing guarantees that there is only one live Atom object for a given
    ``(atom_type, value)`` pair.

    Atom values must be hashable because they are used as part of the cache key.
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
            raise AtomMutationError(msg)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_initialised", False):
            msg = "Atom objects are immutable"
            raise AtomMutationError(msg)
        object.__delattr__(self, name)

    def __repr__(self) -> str:
        return f"{self.atom_type}({self.value!r})"

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


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
