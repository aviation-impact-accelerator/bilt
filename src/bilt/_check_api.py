"""Check consumer source for confidently identifiable uses of Bilt internals."""

import ast
import typing
from contextlib import contextmanager

from bilt._obsolete_api import constructor_message, scan_obsolete_constructors
from bilt.core._validation import Diagnostic

if typing.TYPE_CHECKING:
    import collections.abc

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


class ApiChecker(ast.NodeVisitor):
    """Track explicit imports, straightforward assignments and annotations within scopes."""

    def __init__(self, path: str, package: str, exports: dict[str, str]) -> None:
        self.path = path
        self.package = package
        self.exports = exports
        self.bindings: dict[str, str | None] = {}
        self.diagnostics: list[Diagnostic] = []

    def _report(self, node: ast.expr | ast.stmt, code: str, message: str) -> None:
        self.diagnostics.append(Diagnostic(f"{self.path}:{node.lineno}:{node.col_offset + 1}", message, code))

    def report_obsolete_constructor(self, node: ast.expr, kind: str) -> None:
        self._report(node, "BILT106", constructor_message(kind))

    def _resolve(self, node: ast.expr | None) -> str | None:
        if isinstance(node, ast.Name):
            return self.bindings.get(node.id)
        if isinstance(node, ast.Attribute):
            parent = self._resolve(node.value)
            if parent is not None:
                path = f"{parent}.{node.attr}"
                return self.exports.get(path, path)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = self._annotation(node.left)
            right = self._annotation(node.right)
            if left in _QUANTITIES and right in _QUANTITIES:
                return "bilt.Quantity"
        if (
            isinstance(node, ast.Subscript)
            and self._resolve(node.value) == "typing.Annotated"
            and isinstance(node.slice, ast.Tuple)
            and node.slice.elts
        ):
            return self._annotation(node.slice.elts[0])
        if isinstance(node, ast.Call):
            kind = self._resolve(node.func)
            if kind in _OBJECTS:
                return "bilt.Tensor" if kind == "bilt.tensor" else kind
        return None

    def _annotation(self, node: ast.expr | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                return self._annotation(ast.parse(node.value, mode="eval").body)
            except SyntaxError:
                return None
        kind = self._resolve(node)
        return kind if kind in _OBJECTS else None

    def _import(self, node: ast.stmt, name: str, path: str) -> None:
        if _private_path(path):
            self._report(node, "BILT101", f"Private Bilt import {path!r}; use a documented public import instead.")
        self.bindings[name] = self.exports.get(path, path)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.partition(".")[0]
            self._import(node, name, alias.name)
            if not alias.asname:
                self.bindings[name] = name

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

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._member(node, self._resolve(node.value), node.attr)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "setattr", "delattr", "hasattr"}
            and node.func.id not in self.bindings
            and len(node.args) >= _ATTRIBUTE_ARGUMENTS
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            self._member(node, self._resolve(node.args[0]), node.args[1].value)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        kind = self._resolve(node.value)
        for target in node.targets:
            self.visit(target)
            if isinstance(target, ast.Name):
                self.bindings[target.id] = kind

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            self.bindings[node.target.id] = self._resolve(node.value) or self._annotation(node.annotation)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bindings[node.id] = None

    @contextmanager
    def _scope(self) -> collections.abc.Generator[None]:
        outer = self.bindings
        self.bindings = outer.copy()
        try:
            yield
        finally:
            self.bindings = outer

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        relationship = next(
            (
                kind
                for decorator in node.decorator_list
                if (kind := self._resolve(decorator)) in {"bilt.equation", "bilt.transform"}
            ),
            None,
        )
        self.bindings[node.name] = relationship
        with self._scope():
            arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            for argument in (*arguments, node.args.vararg, node.args.kwarg):
                if argument is not None:
                    self.bindings[argument.arg] = self._annotation(argument.annotation)
            for statement in node.body:
                self.visit(statement)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(expression)
        self.bindings[node.name] = None
        with self._scope():
            for statement in node.body:
                self.visit(statement)


def scan_api(source: str, tree: ast.Module, path: str, package: str, exports: dict[str, str]) -> ApiChecker:
    checker = ApiChecker(path, package, exports)
    checker.visit(tree)
    scan_obsolete_constructors(source, tree, checker.report_obsolete_constructor)
    return checker
