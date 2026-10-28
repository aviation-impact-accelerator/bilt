"""Primitive equation blocks with explicit subsystem ownership and qualified symbols."""

__all__ = ["QualifiedBlock", "QualifiedEquation"]

import dataclasses
import typing

from bilt.core._equation import BasicEquation
from bilt.core._errors import BiltAnalysisError
from bilt.core._structural_analysis._linear_domains import equation_domain_hints, merge_domain_hints
from bilt.symbolic._atoms import SymbolType
from bilt.symbolic._tree import Tree, symbol_tree

if typing.TYPE_CHECKING:
    from bilt.symbolic._domains import Domain


@dataclasses.dataclass(frozen=True)
class QualifiedEquation:
    """One original primitive equation at an implementation instance path.

    Identity is the source object and path, not its display name or expression.
    Only variable symbols and parameter hint names are qualified; operators,
    dimensions, coordinate labels and quantity constants retain their meaning.
    The original equation and its annotations are never rewritten.
    """

    source: BasicEquation
    path: tuple[str, ...] = ()
    tree: Tree = dataclasses.field(init=False, compare=False)
    parameters: tuple[str, ...] = dataclasses.field(init=False, compare=False)
    domain_hints: tuple[tuple[str, Domain], ...] = dataclasses.field(init=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, BasicEquation):
            message = "Qualified equation rows require a primitive equation."
            raise BiltAnalysisError(message)
        if any(not part or "." in part for part in self.path):
            message = "Qualified equation paths require nonempty subsystem names without dots."
            raise BiltAnalysisError(message)
        prefix = ".".join((*self.path, "")) if self.path else ""
        names = {name: prefix + name for name in self.source.__bilt_parameters__}
        object.__setattr__(self, "tree", _qualify_tree(self.source.__bilt_equation_tree__, names))
        object.__setattr__(self, "parameters", tuple(names.values()))
        object.__setattr__(
            self, "domain_hints", tuple((prefix + name, hint) for name, hint in equation_domain_hints(self.source))
        )

    @property
    def label(self) -> str:
        """Return the original source name prefixed by its instance path."""
        return ".".join((*self.path, self.source.__bilt_name__))


@dataclasses.dataclass(frozen=True)
class QualifiedBlock:
    """Explicit primitive rows and unknown columns, before execution planning.

    The caller supplies row and column order. This representation retains
    rectangular or nonlinear formulations without claiming they are solvable.
    Reusing a source in different instances is valid; repeating it at one path
    is a duplicate constraint. No matching, expansion or backend work occurs.
    """

    rows: tuple[QualifiedEquation, ...]
    unknowns: tuple[str, ...]
    domain_hints: tuple[tuple[str, Domain], ...] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if not self.rows or not self.unknowns:
            message = "A qualified block requires primitive rows and unknowns."
            raise BiltAnalysisError(message)
        if len(set(self.rows)) != len(self.rows):
            message = "A qualified block cannot repeat a primitive equation at the same instance path."
            raise BiltAnalysisError(message)
        if len(set(self.unknowns)) != len(self.unknowns):
            message = "A qualified block cannot repeat an unknown."
            raise BiltAnalysisError(message)
        parameters = {name for row in self.rows for name in row.parameters}
        missing = set(self.unknowns) - parameters
        if missing:
            message = f"Qualified block unknowns are not declared by its rows: {sorted(missing)}."
            raise BiltAnalysisError(message)
        hints = merge_domain_hints(hint for row in self.rows for hint in row.domain_hints)
        object.__setattr__(self, "domain_hints", tuple(sorted(hints.items())))

    @property
    def inputs(self) -> frozenset[str]:
        """Return declared row dependencies outside the selected unknown columns."""
        return frozenset(name for row in self.rows for name in row.parameters) - frozenset(self.unknowns)


def _qualify_tree(tree: Tree, names: dict[str, str]) -> Tree:
    """Rewrite each distinct subtree once, preserving shared immutable structure."""
    rewritten: dict[Tree, Tree] = {}

    def visit(node: Tree) -> Tree:
        if node in rewritten:
            return rewritten[node]
        if node.is_leaf and node.head.atom_type is SymbolType:
            name = typing.cast("str", node.head.value)
            if name not in names:
                message = f"Cannot qualify undeclared equation symbol {name!r}."
                raise BiltAnalysisError(message)
            result = symbol_tree(names[name])
        else:
            result = Tree(node.head, tuple(visit(child) for child in node.args))
        rewritten[node] = result
        return result

    return visit(tree)
