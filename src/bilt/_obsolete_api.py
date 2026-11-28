"""Shared recognition of obsolete public constructors."""

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bilt._migration import MigrationContext

if TYPE_CHECKING:
    from collections.abc import Callable

CONSTRUCTORS = {
    "bilt.Quantity": "multiply a number by a unit, e.g. 2.0 * meter",
    "bilt.units.Quantity": "multiply a number by a unit, e.g. 2.0 * meter",
    "bilt.QuantityTensor": "use tensor(..., unit=...)",
    "bilt.Tensor": "use tensor(...)",
    "bilt.Number": "use a plain number or tensor(...)",
    "bilt.Tensor.scalar": "use tensor(value), or tensor(value, unit=...)",
}


def obsolete_constructor(node: ast.expr, context: MigrationContext) -> str | None:
    """Resolve constructor references without inferring the types of values."""
    if not isinstance(node, ast.Call):
        return None
    kind = context.resolve(node.func)
    return kind if kind in CONSTRUCTORS else None


def constructor_message(kind: str) -> str:
    return f"Obsolete constructor {kind}; {CONSTRUCTORS[kind]} instead."


@dataclass(frozen=True, slots=True)
class _ConstructorScan:
    report: Callable[[ast.expr, str], None]

    def apply(self, node: ast.expr, context: MigrationContext) -> None:
        if (kind := obsolete_constructor(node, context)) is not None:
            self.report(node, kind)


def scan_obsolete_constructors(source: str, tree: ast.Module, report: Callable[[ast.expr, str], None]) -> None:
    """Inspect the same lexical scopes as the migration, without proposing edits."""
    MigrationContext(source, tree, (_ConstructorScan(report),)).visit(tree)
