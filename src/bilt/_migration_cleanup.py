"""Conservative, reusable postprocessing of migrated Python source."""

import ast
import re
import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_DYNAMIC = frozenset({"__all__", "globals", "locals", "vars", "eval", "exec", "__import__", "getattr"})


def _references(tree: ast.Module) -> set[str]:
    """Overapproximate references, including quoted annotations and string exports."""
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.update(re.findall(r"\b\w+\b", node.value))
    return names


def _imports(tree: ast.Module) -> set[tuple[str, str]]:
    return {
        (f"{node.module}.{alias.name}", alias.asname or alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None
        for alias in node.names
        if alias.name != "*" and alias.asname != alias.name
    }


def _has_dynamic_access(tree: ast.Module) -> bool:
    return any(
        (isinstance(node, ast.Name) and node.id in _DYNAMIC)
        or (isinstance(node, ast.Attribute) and node.attr in _DYNAMIC)
        or (isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names))
        for node in ast.walk(tree)
    )


def _sole_suite_statements(tree: ast.Module) -> set[ast.stmt]:
    return {
        statements[0]
        for parent in ast.walk(tree)
        if parent is not tree
        for _, statements in ast.iter_fields(parent)
        if isinstance(statements, list) and len(statements) == 1 and isinstance(statements[0], ast.stmt)
    }


def _import_replacement(
    node: ast.ImportFrom,
    lines: list[bytes],
    obsolete: set[tuple[str, str]],
    *,
    sole_statement: bool,
) -> bytes | None:
    """Rewrite a whole, uncommented import statement, retaining its suite."""
    if node.level or node.module is None or node.end_lineno is None or node.end_col_offset is None:
        return None
    first = lines[node.lineno - 1]
    last = lines[node.end_lineno - 1]
    indent = first[: node.col_offset]
    suffix = last[node.end_col_offset :]
    original = b"".join(lines[node.lineno - 1 : node.end_lineno])
    if indent.strip() or suffix.strip() or b"#" in original or b";" in original:
        return None
    remaining = [
        alias
        for alias in node.names
        if alias.asname == alias.name or (f"{node.module}.{alias.name}", alias.asname or alias.name) not in obsolete
    ]
    if len(remaining) == len(node.names):
        return None
    newline = b"\r\n" if last.endswith(b"\r\n") else b"\n" if last.endswith(b"\n") else b""
    if not remaining:
        return indent + b"pass" + newline if sole_statement else b""
    members = ", ".join(
        alias.name if alias.asname is None else f"{alias.name} as {alias.asname}" for alias in remaining
    )
    return indent + f"from {node.module} import {members}".encode() + newline


def remove_obsolete_imports(
    original: str,
    migrated: str,
    removable_imports: frozenset[str],
) -> str:
    """Remove approved from-imports only when a migration removed their last reference.

    This is not a general unused-import fixer. The caller must approve each
    qualified import as safe to remove; ambiguous dynamic access and commented
    statements are left alone.
    """
    if not removable_imports or original == migrated:
        return migrated
    before = ast.parse(original)
    after = ast.parse(migrated)
    if _has_dynamic_access(before) or _has_dynamic_access(after):
        return migrated
    lost = _references(before) - _references(after)
    obsolete = {(path, name) for path, name in _imports(before) if path in removable_imports and name in lost}
    if not obsolete:
        return migrated
    lines = migrated.encode().splitlines(keepends=True)
    sole = _sole_suite_statements(after)
    # Reverse source order keeps all earlier line offsets valid.
    imports = sorted(
        (node for node in ast.walk(after) if isinstance(node, ast.ImportFrom)),
        key=lambda node: (node.lineno, node.col_offset),
        reverse=True,
    )
    for node in imports:
        replacement = _import_replacement(node, lines, obsolete, sole_statement=node in sole)
        if replacement is not None:
            lines[node.lineno - 1 : node.end_lineno] = [replacement]
    result = b"".join(lines).decode()
    ast.parse(result)
    return result


@dataclass(frozen=True, slots=True)
class CleanupOptions:
    """Optional reusable cleanup policy for a registered migration.

    Only imports explicitly approved by the migration are eligible for removal.
    A formatter is an optional caller-provided source-to-source function;
    formatting must preserve the Python AST.
    """

    removable_imports: frozenset[str] = frozenset()
    formatter: Callable[[str], str] | None = None

    def apply(self, original: str, migrated: str, *, path: Path | None = None) -> str:
        if original == migrated:
            return migrated
        if path is None or path.name != "__init__.py":
            migrated = remove_obsolete_imports(original, migrated, self.removable_imports)
        if self.formatter is not None:
            formatted = self.formatter(migrated)
            if ast.dump(ast.parse(migrated)) != ast.dump(ast.parse(formatted)):
                msg = "Migration formatter changed the Python AST"
                raise ValueError(msg)
            migrated = formatted
        return migrated
