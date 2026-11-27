"""Conservative, source-only migration of the pre-protocol quantity API."""

import ast
import difflib
import os
import sys
import typing
from dataclasses import dataclass
from pathlib import Path

from bilt._check_api import ApiChecker

_SKIP = frozenset({".venv", "venv", "build", "dist", "node_modules", "__pycache__", "site-packages"})
_QUANTITY = frozenset({"bilt.Quantity", "bilt.units.Quantity"})
_TENSOR = "bilt.QuantityTensor"
_PAIR = 2


@dataclass(frozen=True, slots=True)
class Edit:
    start: int
    end: int
    replacement: bytes


@dataclass(frozen=True, slots=True)
class Migration:
    source: str
    notes: tuple[str, ...]


class QuantityMigration(ApiChecker):
    """Use the checker's name resolution, but edit only unambiguous source spans."""

    def __init__(self, source: str) -> None:
        super().__init__("", "", {})
        self.source = source.encode()
        self.offsets = [0]
        for line in self.source.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.edits: list[Edit] = []
        self.notes: list[str] = []

    def _span(self, node: ast.expr) -> tuple[int, int]:
        assert node.end_lineno is not None  # noqa: S101
        assert node.end_col_offset is not None  # noqa: S101
        return (
            self.offsets[node.lineno - 1] + node.col_offset,
            self.offsets[node.end_lineno - 1] + node.end_col_offset,
        )

    def _text(self, node: ast.expr) -> bytes:
        start, end = self._span(node)
        return self.source[start:end]

    def _replace(self, node: ast.expr, replacement: bytes) -> bool:
        original = self._text(node)
        if b"#" in original or b"\n" in original or b"\r" in original:
            self.notes.append(f"{node.lineno}: review multiline or commented expression manually")
            return False
        start, end = self._span(node)
        self.edits.append(Edit(start, end, replacement))
        return True

    @typing.override
    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.BitOr):
            left = self._constructor_reference(node.left)
            right = self._constructor_reference(node.right)
            if left in _QUANTITY and right == _TENSOR:
                if self._replace(node, self._text(node.left)):
                    return
            elif right in _QUANTITY and left == _TENSOR and self._replace(node, self._text(node.right)):
                return
        self.generic_visit(node)

    def _factory(self) -> bytes | None:
        for name, kind in self.bindings.items():
            if kind == "bilt" and name.isidentifier():
                return f"{name}.tensor".encode()
        return None

    def _migrate_call(self, node: ast.Call, kind: str) -> bool:
        if kind in _QUANTITY and len(node.args) == _PAIR and not node.keywords:
            value, unit = node.args
            if not isinstance(value, ast.Starred) and not isinstance(unit, ast.Starred):
                return self._replace(node, b"((" + self._text(value) + b") * (" + self._text(unit) + b"))")
        if (
            kind == "bilt.Tensor.scalar"
            and len(node.args) == 1
            and not isinstance(node.args[0], ast.Starred)
            and all(kw.arg == "unit" for kw in node.keywords)
            and (factory := self._factory()) is not None
        ):
            return self._replace(node.func, factory)
        return False

    @typing.override
    def visit_Call(self, node: ast.Call) -> None:
        kind = self._constructor_reference(node.func)
        if kind is not None:
            if self._migrate_call(node, kind):
                return
            self.notes.append(f"{node.lineno}: review obsolete {kind} constructor manually")
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) == _PAIR
            and self._constructor_reference(node.args[1]) == "bilt.Tensor"
        ):
            self.notes.append(f"{node.lineno}: review Tensor runtime check for unit-bearing values")
        super().visit_Call(node)

    @typing.override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # The checker is flow-sensitive, not a full lexical scope analyser.
        # Skip bodies that can shadow imported names before their assignment.
        if any(
            isinstance(item, (ast.Name, ast.Global, ast.Nonlocal))
            and (not isinstance(item, ast.Name) or isinstance(item.ctx, (ast.Store, ast.Del)))
            for item in ast.walk(node)
        ):
            self.notes.append(f"{node.lineno}: review function with local bindings manually")
            return
        super().visit_FunctionDef(node)

    @typing.override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.notes.append(f"{node.lineno}: review asynchronous function manually")

    @typing.override
    def visit_Lambda(self, _node: ast.Lambda) -> None:
        return

    @typing.override
    def visit_ListComp(self, _node: ast.ListComp) -> None:
        return

    @typing.override
    def visit_SetComp(self, _node: ast.SetComp) -> None:
        return

    @typing.override
    def visit_DictComp(self, _node: ast.DictComp) -> None:
        return

    @typing.override
    def visit_GeneratorExp(self, _node: ast.GeneratorExp) -> None:
        return


def migrate_source(source: str) -> Migration:
    """Rewrite a source string without importing or executing the model."""
    visitor = QuantityMigration(source)
    visitor.visit(ast.parse(source))
    result = visitor.source
    boundary = len(result)
    for edit in sorted(visitor.edits, key=lambda item: item.start, reverse=True):
        if edit.end > boundary:
            msg = "Overlapping migration edits"
            raise ValueError(msg)
        result = result[: edit.start] + edit.replacement + result[edit.end :]
        boundary = edit.start
    migrated = result.decode()
    ast.parse(migrated)
    return Migration(migrated, tuple(dict.fromkeys(visitor.notes)))


def _files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    files: list[Path] = []
    for parent, directories, names in os.walk(target):
        directories[:] = sorted(
            name
            for name in directories
            if name not in _SKIP and not name.startswith(".") and not (Path(parent) / name).is_symlink()
        )
        files.extend(
            Path(parent) / name
            for name in sorted(names)
            if name.endswith(".py") and not (Path(parent) / name).is_symlink()
        )
    return files


def migrate(target: Path, *, write: bool) -> int:
    """Preview or apply source edits. Return 1 if manual review remains."""
    if target.is_symlink() or not target.exists() or (target.is_file() and target.suffix != ".py"):
        sys.stderr.write(f"{target}: expected an existing Python file or directory, not a symlink\n")
        return 2
    changed = 0
    unresolved = 0
    for path in _files(target):
        try:
            source = path.read_bytes().decode("utf-8")
            result = migrate_source(source)
            if source != result.source:
                changed += 1
                if write:
                    path.write_bytes(result.source.encode())
                else:
                    sys.stdout.writelines(
                        difflib.unified_diff(
                            source.splitlines(keepends=True),
                            result.source.splitlines(keepends=True),
                            fromfile=str(path),
                            tofile=str(path),
                        )
                    )
            for note in result.notes:
                sys.stderr.write(f"{path}:{note}\n")
            unresolved += len(result.notes)
        except (OSError, UnicodeError, SyntaxError, ValueError) as error:
            sys.stderr.write(f"{path}: {error}\n")
            unresolved += 1
    action = "Updated" if write else "Would update"
    sys.stdout.write(f"{action} {changed} file(s); {unresolved} manual review item(s).\n")
    return int(bool(unresolved))
