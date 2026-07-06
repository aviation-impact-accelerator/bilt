from __future__ import annotations

import typing
import weakref

from bilt.symbolic._atoms import (
    EQUALS,
    Atom,
    RelationOpType,
    floating,
    integer,
    relation_operator,
    symbol,
    symbol_domain,
)
from bilt.symbolic._domains import Domain, DomainLike, as_domain
from bilt.symbolic._errors import TreeMutationError

if typing.TYPE_CHECKING:
    from bilt.symbolic._dim import Dim

__all__ = [
    "Tree",
    "equality_tree",
    "float_tree",
    "int_tree",
    "relation_tree",
    "symbol_tree",
    "tree_hash",
]


_tree_cache: weakref.WeakValueDictionary[
    tuple[Atom, tuple[Tree, ...], Domain],
    Tree,
] = weakref.WeakValueDictionary()

_NUMBER_OF_SUB_TREES = 2


@typing.final
class Tree:
    """Hash-consed immutable symbolic tree.

    Leaves are represented as:

        Tree(symbol("x"))

    Domain-aware leaves can also be represented as:

        Tree(symbol("x", domain=domain(time)))

    Operations are represented as:

        Tree(ADD, (x, y))

    Relations are represented in the same structure:

        Tree(EQUALS, (lhs, rhs))
        Tree(LESS_THAN, (lhs, rhs))
        Tree(LESS_THAN_OR_EQUALS, (lhs, rhs))
    """

    __slots__ = (
        "__weakref__",
        "_locked",
        "args",
        "domain",
        "head",
    )

    head: Atom
    args: tuple[Tree, ...]
    domain: Domain

    def __new__(
        cls,
        head: Atom,
        args: tuple[Tree, ...] = (),
        *,
        domain: DomainLike | None = None,
    ) -> Tree:
        tree_domain = _infer_domain(head, args, domain)
        key = (head, args, tree_domain)

        if (existing := _tree_cache.get(key)) is not None:
            return existing

        obj = object.__new__(cls)
        object.__setattr__(obj, "head", head)
        object.__setattr__(obj, "args", args)
        object.__setattr__(obj, "domain", tree_domain)
        object.__setattr__(obj, "_locked", True)

        return _tree_cache.setdefault(key, obj)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            msg = f"{self.__class__.__name__} is immutable. This cannot be updated once it is set."
            raise TreeMutationError(msg)

        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_locked", False):
            msg = f"{self.__class__.__name__} is immutable. It cannot be deleted once it is set."
            raise TreeMutationError(msg)

        object.__delattr__(self, name)

    @property
    def is_leaf(self) -> bool:
        """True when this node carries a value rather than an operation."""
        return not self.args

    @property
    def is_relation(self) -> bool:
        """True when this tree represents a symbolic relation."""
        return self.head.atom_type is RelationOpType

    @property
    def is_equality(self) -> bool:
        """True when this tree represents symbolic equality."""
        return self.head is EQUALS

    @property
    def operator(self) -> Atom:
        """The relation operator for a relation tree."""
        if not self.is_relation:
            msg = "Tree is not a relation. Tree head node must be a relation operator atom."
            raise ValueError(msg)

        return self.head

    @property
    def left(self) -> Tree:
        """Left-hand side of a binary relation tree."""
        if not self.is_relation or len(self.args) != _NUMBER_OF_SUB_TREES:
            msg = "Tree is not a binary relation and has no left-hand side."
            raise ValueError(msg)

        return self.args[0]

    @property
    def right(self) -> Tree:
        """Right-hand side of a binary relation tree."""
        if not self.is_relation or len(self.args) != _NUMBER_OF_SUB_TREES:
            msg = "Tree is not a binary relation and has no right-hand side."
            raise ValueError(msg)

        return self.args[1]

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        if self.is_leaf:
            if self.domain == Domain.scalar():
                return f"{self.__class__.__name__}({self.head!r})"

            return f"{self.__class__.__name__}({self.head!r}, domain={self.domain!r})"

        if self.domain == Domain.scalar():
            return f"{self.__class__.__name__}({self.head!r}, {self.args!r})"

        return f"{self.__class__.__name__}({self.head!r}, {self.args!r}, domain={self.domain!r})"

    def __str__(self) -> str:
        if self.is_leaf:
            return str(self.head)

        if self.is_relation and len(self.args) == _NUMBER_OF_SUB_TREES:
            return f"({self.args[0]} {self.head} {self.args[1]})"

        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.head}({args_str})"


def symbol_tree(
    name: str,
    *,
    domain: DomainLike | None = None,
) -> Tree:
    return Tree(symbol(name, domain=domain))


def int_tree(
    n: int,
    *,
    domain: DomainLike | None = None,
) -> Tree:
    return Tree(integer(n), domain=domain)


def float_tree(
    x: float,
    *,
    domain: DomainLike | None = None,
) -> Tree:
    return Tree(floating(x), domain=domain)


def relation_tree(
    operator: Atom | str,
    left: Tree,
    right: Tree,
    *,
    domain: DomainLike | None = None,
) -> Tree:
    """Construct a binary symbolic relation tree.

    Examples:
        relation_tree("==", x, y)
        relation_tree(EQUALS, x, y)
    """
    if isinstance(operator, str):
        operator = relation_operator(operator)

    if operator.atom_type is not RelationOpType:
        msg = f"Expected relation operator atom, got {operator!r}."
        raise TypeError(msg)

    return Tree(operator, (left, right), domain=domain)


def equality_tree(
    left: Tree,
    right: Tree,
    *,
    domain: DomainLike | None = None,
) -> Tree:
    """Construct a symbolic equality tree."""
    return relation_tree(EQUALS, left, right, domain=domain)


def tree_hash(tree: Tree) -> int:
    return hash(
        (
            tree.head,
            tree.domain,
            tuple(tree_hash(child) for child in tree.args),
        )
    )


def _infer_domain(
    head: Atom,
    args: tuple[Tree, ...],
    explicit_domain: DomainLike | None,
) -> Domain:
    if explicit_domain is not None:
        return as_domain(explicit_domain)

    if not args:
        return symbol_domain(head)

    return _merge_domains(*(arg.domain for arg in args))


def _merge_domains(*domains: Domain) -> Domain:
    dims: list[Dim] = []

    for current_domain in domains:
        for current_dim in current_domain.dims:
            if current_dim not in dims:
                dims.append(current_dim)

    return Domain(tuple(dims))
