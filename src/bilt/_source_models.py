"""Shared recognition of directly imported Bilt model declarations."""

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

DECORATORS = frozenset({"system", "equation", "transform"})
IMPLEMENTATIONS = frozenset({"Implementation"})


def bilt_imports(statements: Iterable[ast.AST]) -> tuple[set[str], set[str], set[str]]:
    """Recognise explicitly imported public Bilt decorators and constructors."""
    decorators: set[str] = set()
    implementations: set[str] = set()
    modules: set[str] = set()
    for statement in statements:
        if isinstance(statement, ast.ImportFrom) and statement.module == "bilt":
            for alias in statement.names:
                local = alias.asname or alias.name
                if alias.name in DECORATORS:
                    decorators.add(local)
                elif alias.name == "Implementation":
                    implementations.add(local)
        elif isinstance(statement, ast.Import):
            modules.update(alias.asname or "bilt" for alias in statement.names if alias.name == "bilt")
    return decorators, implementations, modules


def recognised(function: ast.expr, names: set[str], modules: set[str], *, members: frozenset[str]) -> bool:
    if isinstance(function, ast.Name):
        return function.id in names
    return (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id in modules
        and function.attr in members
    )
