"""Check consumer source for confidently identifiable uses of Bilt internals."""

import ast
import typing
from collections import Counter

from bilt._lexical import LexicalVisitor
from bilt._obsolete_api import constructor_message, scan_obsolete_constructors
from bilt.core._validation import Diagnostic

if typing.TYPE_CHECKING:
    import symtable

_OBJECTS = frozenset(
    {
        "bilt.Implementation",
        "bilt.Number",
        "bilt.Quantity",
        "bilt.units.Quantity",
        "bilt.QuantityTensor",
        "bilt.Tensor",
        "bilt.PYTHON_BACKEND",
        "bilt.JAX_BACKEND",
        "bilt.equation",
        "bilt.transform",
        "bilt.tensor",
    }
)
_IMPLEMENTATION_METHODS = frozenset({"evaluate", "get_evaluation", "analyser_for", "structural_analysis"})
_BACKEND_METHODS = frozenset({"transpile", "chain_case", "apply_alignment", "transpile_linear_solve"})
_BACKENDS = frozenset({"bilt.PYTHON_BACKEND", "bilt.JAX_BACKEND"})
_ATTRIBUTE_ARGUMENTS = 2
_QUANTITIES = frozenset({"bilt.Quantity", "bilt.units.Quantity", "bilt.QuantityTensor"})


def _private_path(path: str) -> bool:
    return path.startswith("bilt.") and (
        path == "bilt.core"
        or path.startswith("bilt.core.")
        or any(part.startswith("_") for part in path.split(".")[1:])
    )


def _stable_module_values(tree: ast.Module) -> set[str]:
    candidates: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            candidates.update(target.id for target in statement.targets if isinstance(target, ast.Name))
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            candidates.add(statement.target.id)
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            candidates.add(statement.name)
    writes: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            writes[node.id] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            writes[node.name] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            writes.update(alias.asname or alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.Global):
            writes.update(node.names)
    return {name for name in candidates if writes[name] == 1}


class ApiChecker(LexicalVisitor):
    """Track explicit imports, straightforward assignments and annotations within scopes."""

    def __init__(self, source: str, tree: ast.Module, path: str, package: str, exports: dict[str, str]) -> None:
        super().__init__(source, tree, package=package)
        self.path = path
        self.exports = exports
        self.module_bindings = self.bindings
        self.stable_values = _stable_module_values(tree)
        self.diagnostics: list[Diagnostic] = []

    def _report(self, node: ast.expr | ast.stmt, code: str, message: str) -> None:
        self.diagnostics.append(Diagnostic(f"{self.path}:{node.lineno}:{node.col_offset + 1}", message, code))

    def report_obsolete_constructor(self, node: ast.expr, kind: str) -> None:
        self._report(node, "BILT106", constructor_message(kind))

    @typing.override
    def resolve(self, node: ast.expr | None) -> str | None:
        """Resolve references and infer only explicitly recognised Bilt object types."""
        if (reference := super().resolve(node)) is not None:
            return self.exports.get(reference, reference)
        if isinstance(node, ast.Name) and node.id not in self.bindings:
            members = self.parameter_members.get(node.id, set())
            if len(members) == 1:
                return next(iter(members))
            if members and members <= _QUANTITIES:
                return "bilt.Quantity"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = self.annotation_binding(node.left)
            right = self.annotation_binding(node.right)
            if left in _QUANTITIES and right in _QUANTITIES:
                return "bilt.Quantity"
        if (
            isinstance(node, ast.Subscript)
            and self.resolve(node.value) == "typing.Annotated"
            and isinstance(node.slice, ast.Tuple)
            and node.slice.elts
        ):
            return self.annotation_binding(node.slice.elts[0])
        if isinstance(node, ast.Call):
            kind = self.resolve(node.func)
            if kind in _OBJECTS:
                return "bilt.Tensor" if kind == "bilt.tensor" else kind
        return None

    @typing.override
    def annotation_binding(self, node: ast.expr | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                return self.annotation_binding(ast.parse(node.value, mode="eval").body)
            except SyntaxError:
                return None
        kind = self.resolve(node)
        return kind if kind in _OBJECTS else None

    def _import(self, node: ast.stmt, name: str, path: str) -> None:
        if _private_path(path):
            self._report(node, "BILT101", f"Private Bilt import {path!r}; use a documented public import instead.")
        self.bindings[name] = self.exports.get(path, path)

    @typing.override
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.partition(".")[0]
            self._import(node, name, alias.name)
            if not alias.asname:
                self.bindings[name] = name

    @typing.override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level:
            parts = self.package.split(".") if self.package else []
            if node.level > len(parts):
                return
            module = ".".join((*parts[: len(parts) - node.level + 1], *([module] if module else [])))
        for alias in node.names:
            self._import(node, alias.asname or alias.name, f"{module}.{alias.name}")

    def _member(self, node: ast.expr, parent: str | None, name: str) -> None:
        if name.startswith("__bilt_"):
            self._report(node, "BILT103", f"Reserved Bilt metadata {name!r}; use declared system fields instead.")
        elif parent == "bilt.Implementation" and name in _IMPLEMENTATION_METHODS:
            self._report(
                node,
                "BILT104",
                f"Internal Implementation.{name}; activate the implementation and request a declared system field.",
            )
        elif parent in _BACKENDS and name in _BACKEND_METHODS:
            self._report(
                node, "BILT105", f"Internal backend method {name!r}; select a backend with its context manager."
            )
        elif parent in _OBJECTS and name.startswith("_") and not name.startswith("__"):
            self._report(node, "BILT102", f"Private Bilt attribute {name!r}; use the documented public API.")
        elif (
            parent is not None
            and parent not in _OBJECTS
            and not _private_path(parent)
            and _private_path(f"{parent}.{name}")
        ):
            self._report(node, "BILT101", f"Private Bilt namespace or attribute {name!r}; use a public import instead.")

    @typing.override
    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._member(node, self.resolve(node.value), node.attr)
        self.generic_visit(node)

    @typing.override
    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "setattr", "delattr", "hasattr"}
            and self.builtin_visible(node.func.id)
            and len(node.args) >= _ATTRIBUTE_ARGUMENTS
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            self._member(node, self.resolve(node.args[0]), node.args[1].value)
        self.generic_visit(node)

    @typing.override
    def function_binding(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
        return next(
            (
                kind
                for decorator in node.decorator_list
                if (kind := self.resolve(decorator)) in {"bilt.equation", "bilt.transform"}
            ),
            None,
        )

    @typing.override
    def function_bindings(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda, table: symtable.SymbolTable
    ) -> dict[str, str | None]:
        bindings = super().function_bindings(node, table)
        bindings.update(
            {
                name: self.module_bindings[name]
                for name in self.stable_values.intersection(self.module_bindings)
                if name not in table.get_identifiers() or table.lookup(name).is_global()
            }
        )
        return bindings

    @typing.override
    def local_names(self, node: ast.FunctionDef | ast.AsyncFunctionDef, table: symtable.SymbolTable) -> set[str]:
        return super().local_names(node, table) | {
            name
            for name in table.get_identifiers()
            if table.lookup(name).is_local() and table.lookup(name).is_assigned() and not table.lookup(name).is_imported()
        }


def scan_api(source: str, tree: ast.Module, path: str, package: str, exports: dict[str, str]) -> ApiChecker:
    checker = ApiChecker(source, tree, path, package, exports)
    checker.visit(tree)
    scan_obsolete_constructors(source, tree, checker.report_obsolete_constructor)
    return checker
