"""Shared recognition of obsolete public constructors."""

import ast
import typing

from bilt._lexical import LexicalVisitor

if typing.TYPE_CHECKING:
    from collections.abc import Callable

CONSTRUCTORS = {
    "bilt.Quantity": "multiply a number by a unit, e.g. 2.0 * meter",
    "bilt.units.Quantity": "multiply a number by a unit, e.g. 2.0 * meter",
    "bilt.QuantityTensor": "use tensor(..., unit=...)",
    "bilt.Tensor": "use tensor(...)",
    "bilt.Number": "use a plain number or tensor(...)",
    "bilt.Tensor.scalar": "use tensor(value), or tensor(value, unit=...)",
}


def obsolete_constructor(node: ast.expr, context: LexicalVisitor) -> str | None:
    """Resolve constructor references without inferring the types of values."""
    if not isinstance(node, ast.Call):
        return None
    kind = context.resolve(node.func)
    return kind if kind in CONSTRUCTORS else None


def constructor_message(kind: str) -> str:
    return f"Obsolete constructor {kind}; {CONSTRUCTORS[kind]} instead."


class _ConstructorScan(LexicalVisitor):
    def __init__(self, source: str, tree: ast.Module, report: Callable[[ast.expr, str], None]) -> None:
        super().__init__(source, tree)
        self.report = report

    @typing.override
    def visit(self, node: ast.AST) -> None:
        super().visit(node)
        if isinstance(node, ast.expr) and (kind := obsolete_constructor(node, self)) is not None:
            self.report(node, kind)


def scan_obsolete_constructors(source: str, tree: ast.Module, report: Callable[[ast.expr, str], None]) -> None:
    """Inspect the same lexical scopes as the migration, without proposing edits."""
    _ConstructorScan(source, tree, report).visit(tree)
