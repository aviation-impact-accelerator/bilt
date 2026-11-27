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
_DIMENSIONS = frozenset(
    "AMOUNT AREA CHARGE CONDUCTANCE CURRENCY CURRENT DIMENSIONLESS DYNAMIC_VISCOSITY ENERGY FORCE "
    "LENGTH LUMINOSITY MASS POWER PRESSURE RESISTANCE TEMPERATURE TIME VOLTAGE VOLUME".split()
)


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

    def __init__(self, source: str, tree: ast.Module) -> None:
        super().__init__("", "", {})
        self.source = source.encode()
        self.offsets = [0]
        for line in self.source.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.edits: list[Edit] = []
        self.notes: list[str] = []
        self.imports: set[str] = set()
        self.tree = tree
        self.names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.names.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
        self.names.update(
            node.asname or node.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.alias)
        )
        self.names.update(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )

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

    def _replace(self, node: ast.expr, replacement: bytes, *, multiline: bool = False) -> bool:
        original = self._text(node)
        if b"#" in original or (not multiline and (b"\n" in original or b"\r" in original)):
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
                if self._replace(node, self._text(node.left), multiline=True):
                    return
            elif right in _QUANTITY and left == _TENSOR and self._replace(node, self._text(node.right), multiline=True):
                return
        self.generic_visit(node)

    def _public_name(self, member: str) -> bytes:
        for name, kind in self.bindings.items():
            if kind == f"bilt.{member}":
                return name.encode()
            if kind == "bilt":
                return f"{name}.{member}".encode()
        # Reserve a globally unused identifier, including names in nested scopes.
        name = member
        while name in self.names:
            name = "_" + name
        self.imports.add(member if name == member else f"{member} as {name}")
        return name.encode()

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
        source = ("from bilt import " + ", ".join(sorted(self.imports))).encode() + newline
        return Edit(offset, offset, source)

    def _dimension(self, node: ast.expr) -> bool:
        path = self._resolve(node)
        return (
            path is not None
            and path.startswith("bilt.units.dimensions.")
            and path.rsplit(".", 1)[-1] in _DIMENSIONS
        )

    @typing.override
    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            self._resolve(node.value) == "typing.Annotated"
            and isinstance(node.slice, ast.Tuple)
            and node.slice.elts
        ):
            base, *metadata = node.slice.elts
            if self._constructor_reference(base) == "bilt.Tensor" and any(map(self._dimension, metadata)):
                self._replace(base, self._public_name("QuantityTensor"))
        self.generic_visit(node)

    def _arguments(self, node: ast.Call, parameters: tuple[str, ...]) -> dict[str, ast.expr] | None:
        if len(node.args) > len(parameters) or any(isinstance(arg, ast.Starred) for arg in node.args):
            return None
        result = dict(zip(parameters, node.args, strict=False))
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg not in parameters or keyword.arg in result:
                return None
            result[keyword.arg] = keyword.value
        return result

    def _quantity_call(self, node: ast.Call) -> bool:
        args = self._arguments(node, ("value", "unit"))
        # Keep evaluation order: unit= before value= may have side effects.
        if args is None or tuple(args) != ("value", "unit"):
            return False
        value, unit = args.values()
        return self._replace(node, b"((" + self._text(value) + b") * (" + self._text(unit) + b"))")

    def _tensor_call(self, node: ast.Call, kind: str) -> bool:
        scalar = kind == "bilt.Tensor.scalar"
        parameters = ("value", "unit") if scalar else ("values", "domain", "unit")
        args = self._arguments(node, parameters)
        if args is None or parameters[0] not in args or (not scalar and "domain" not in args):
            return False
        if kind == "bilt.QuantityTensor" and "unit" not in args:
            return False
        # Only change the callee and positional argument labels when possible.
        # Rebuilding a commented or multiline call remains a manual operation.
        if any(mark in self._text(node) for mark in (b"#", b"\n", b"\r")):
            return False
        factory = self._public_name("tensor")
        parts = [
            (("values" if name == "value" else name).encode() + b"=" + self._text(arg))
            for name, arg in args.items()
        ]
        if scalar and tuple(args)[0] == "value":
            parts[0] = self._text(args["value"])
        return self._replace(node, factory + b"(" + b", ".join(parts) + b")")

    def _migrate_call(self, node: ast.Call, kind: str) -> bool:
        if kind in _QUANTITY:
            return self._quantity_call(node)
        if kind in {"bilt.Tensor", "bilt.QuantityTensor", "bilt.Tensor.scalar"}:
            return self._tensor_call(node, kind)
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
    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    @typing.override
    def visit_ListComp(self, node: ast.ListComp) -> None:
        return

    @typing.override
    def visit_SetComp(self, node: ast.SetComp) -> None:
        return

    @typing.override
    def visit_DictComp(self, node: ast.DictComp) -> None:
        return

    @typing.override
    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        return


def migrate_source(source: str) -> Migration:
    """Rewrite a source string without importing or executing the model."""
    tree = ast.parse(source)
    visitor = QuantityMigration(source, tree)
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
