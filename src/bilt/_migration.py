"""Internal rule-driven framework for source-only Python API migrations."""

import ast
import difflib
import sys
import typing
from dataclasses import dataclass

from bilt._lexical import LexicalVisitor
from bilt._migration_expressions import binary_expression, contextual_expression
from bilt._source_project import source_files

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from bilt._migration_cleanup import CleanupOptions

@dataclass(frozen=True, slots=True)
class Edit:
    start: int
    end: int
    replacement: bytes


@dataclass(frozen=True, slots=True)
class Migration:
    source: str
    notes: tuple[str, ...]


class MigrationRule(typing.Protocol):
    """Independent source rewrite or review rule, applied after child nodes."""

    def apply(self, node: ast.expr, context: MigrationContext) -> None:
        """Inspect an expression and optionally propose edits or review notes."""
        ...


class MigrationContext(LexicalVisitor):
    """Compose source edits for registered rules using lexical analysis."""

    def __init__(self, source: str, tree: ast.Module, rules: tuple[MigrationRule, ...]) -> None:
        super().__init__(source, tree)
        self.rules = rules
        self.source = source.encode()
        self.offsets = [0]
        for line in self.source.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.edits: list[Edit] = []
        self.notes: list[str] = []
        self.imports: dict[tuple[str, str], str] = {}

    @typing.override
    def visit(self, node: ast.AST) -> None:
        super().visit(node)
        if isinstance(node, ast.expr):
            for rule in self.rules:
                rule.apply(node, self)

    def note(self, node: ast.expr, message: str) -> None:
        """Record a source-located manual review item."""
        self.notes.append(f"{node.lineno}: {message}")

    @typing.override
    def unresolved_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.notes.append(f"{node.lineno}: review unresolved lexical scope manually")

    def _span(self, node: ast.expr) -> tuple[int, int]:
        assert node.end_lineno is not None  # noqa: S101
        assert node.end_col_offset is not None  # noqa: S101
        return (
            self.offsets[node.lineno - 1] + node.col_offset,
            self.offsets[node.end_lineno - 1] + node.end_col_offset,
        )

    def text(self, node: ast.expr) -> bytes:
        """Return original UTF-8 source bytes of an expression."""
        start, end = self._span(node)
        return self.source[start:end]

    def render(self, node: ast.expr) -> bytes:
        """Render a source span including already planned edits to its children."""
        start, end = self._span(node)
        result = self.source[start:end]
        for edit in sorted(self.edits, key=lambda item: item.start, reverse=True):
            if start <= edit.start and edit.end <= end:
                result = result[: edit.start - start] + edit.replacement + result[edit.end - start :]
        return result

    def replace(self, node: ast.expr, replacement: bytes, *, multiline: bool = False) -> bool:
        """Replace an expression without discarding comments."""
        original = self.text(node)
        if b"#" in original or (not multiline and (b"\n" in original or b"\r" in original)):
            self.notes.append(f"{node.lineno}: review multiline or commented expression manually")
            return False
        start, end = self._span(node)
        self.edits = [edit for edit in self.edits if not (start <= edit.start and edit.end <= end)]
        self.edits.append(Edit(start, end, replacement))
        return True

    def replace_expression(self, node: ast.expr, replacement: bytes) -> bool:
        """Replace an expression, adding grouping only where the grammar requires it."""
        start, end = self._span(node)
        try:
            rendered = contextual_expression(self.source, start, end, replacement)
        except ValueError as error:
            self.note(node, str(error))
            return False
        return self.replace(node, rendered, multiline=True)

    def replace_binary(self, node: ast.expr, left: ast.expr, operator: str, right: ast.expr) -> bool:
        """Replace an expression with a binary operation, preserving nested edits."""
        try:
            replacement = binary_expression(self.render(left), operator, self.render(right))
        except ValueError as error:
            self.note(node, str(error))
            return False
        return self.replace_expression(node, replacement)

    def ensure_import(self, module: str, member: str) -> bytes:
        """Return a usable imported reference, reserving a fresh alias if necessary."""
        for name, kind in self.bindings.items():
            if self.allowed is not None and name not in self.allowed:
                continue
            if kind == f"{module}.{member}":
                return name.encode()
            if kind == module:
                return f"{name}.{member}".encode()
        key = (module, member)
        if key not in self.imports:
            name = member
            while name in self.names:
                name = "_" + name
            self.names.add(name)
            self.imports[key] = name
        return self.imports[key].encode()

    def import_edit(self) -> Edit | None:
        if not self.imports:
            return None
        body = self.tree.body
        index = 0
        if ast.get_docstring(self.tree) is not None:
            index = 1
        while index < len(body):
            statement = body[index]
            if not isinstance(statement, ast.ImportFrom) or statement.module != "__future__":
                break
            index += 1
        if index == len(body):
            return None
        offset = self.offsets[body[index].lineno - 1]
        if index and body[index - 1].end_lineno == body[index].lineno:
            msg = "Cannot insert imports after a same-line docstring or future import"
            raise ValueError(msg)
        newline = b"\r\n" if b"\r\n" in self.source else b"\n"
        source = b"".join(
            (f"from {module} import {member}" + (f" as {name}" if member != name else "")).encode() + newline
            for (module, member), name in sorted(self.imports.items())
        )
        return Edit(offset, offset, source)


def migrate_source(
    source: str,
    rules: tuple[MigrationRule, ...],
    *,
    cleanup: CleanupOptions | None = None,
    path: Path | None = None,
    review: Callable[[str, str], tuple[str, ...]] | None = None,
) -> Migration:
    """Rewrite a source string without importing or executing the model."""
    tree = ast.parse(source)
    visitor = MigrationContext(source, tree, rules)
    visitor.visit(tree)
    if (edit := visitor.import_edit()) is not None:
        visitor.edits.append(edit)
    result = visitor.source
    boundary = len(result)
    for edit in sorted(visitor.edits, key=lambda item: (item.start, item.end), reverse=True):
        if edit.end > boundary:
            msg = "Overlapping migration edits"
            raise ValueError(msg)
        result = result[: edit.start] + edit.replacement + result[edit.end :]
        boundary = edit.start
    migrated = result.decode()
    ast.parse(migrated)
    if cleanup is not None and not visitor.notes:
        migrated = cleanup.apply(source, migrated, path=path)
    if review is not None:
        visitor.notes.extend(review(source, migrated))
    return Migration(migrated, tuple(dict.fromkeys(visitor.notes)))


def migrate(
    target: Path,
    rules: tuple[MigrationRule, ...],
    *,
    write: bool,
    root: Path | None = None,
    excludes: Sequence[str] = (),
    cleanup: CleanupOptions | None = None,
    review: Callable[[str, str], tuple[str, ...]] | None = None,
) -> int:
    """Preview or apply source edits. Return 1 if manual review remains."""
    if target.is_symlink() or not target.exists() or (target.is_file() and target.suffix != ".py"):
        sys.stderr.write(f"{target}: expected an existing Python file or directory, not a symlink\n")
        return 2
    target = target.resolve()
    root = root.resolve() if root is not None else (target if target.is_dir() else target.parent)
    if not target.is_relative_to(root):
        sys.stderr.write(f"{target}: expected a source inside {root}\n")
        return 2
    changed = 0
    unresolved = 0
    for path in source_files(target, root, excludes):
        try:
            source = path.read_bytes().decode("utf-8")
            result = migrate_source(source, rules, cleanup=cleanup, path=path, review=review)
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
