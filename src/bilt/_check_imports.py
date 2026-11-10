"""Recover from errors in recognised top-level Bilt definitions during imports.

Ordinary statements and non-source loaders keep Python's normal import behaviour.
"""

import ast
import contextlib
import dataclasses
import importlib.abc
import importlib.machinery
import sys
import typing
from pathlib import Path

from bilt._base_error import BiltError
from bilt.core._errors import BiltCollectedValidationError, BiltUnresolvedAnnotationError
from bilt.core._validation import Diagnostic, code_for

if typing.TYPE_CHECKING:
    import collections.abc
    import types

_REPORT = "__bilt_check_report__"
_SKIP = "__bilt_check_skip__"
_ERROR = "__bilt_check_error_type__"
_CAUGHT = "__bilt_check_caught__"
_DECORATORS = frozenset({"system", "equation", "transform"})


@dataclasses.dataclass(frozen=True, slots=True)
class SkippedDefinition:
    """A definition not checked because one of its dependencies failed."""

    location: str
    definition: str
    dependency: str


def _bilt_imports(tree: ast.Module) -> tuple[set[str], set[str], set[str]]:
    """Recognise explicitly imported public Bilt decorators and constructors."""
    decorators: set[str] = set()
    implementations: set[str] = set()
    modules: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom) and statement.module == "bilt":
            for alias in statement.names:
                local = alias.asname or alias.name
                if alias.name in _DECORATORS:
                    decorators.add(local)
                elif alias.name == "Implementation":
                    implementations.add(local)
        elif isinstance(statement, ast.Import):
            modules.update(alias.asname or "bilt" for alias in statement.names if alias.name == "bilt")
    return decorators, implementations, modules


def _recognised(function: ast.expr, names: set[str], modules: set[str], *, members: frozenset[str]) -> bool:
    if isinstance(function, ast.Name):
        return function.id in names
    return (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id in modules
        and function.attr in members
    )


def _definition_names(
    statement: ast.stmt, decorators: set[str], implementations: set[str], modules: set[str]
) -> tuple[str, ...] | None:
    """Return names defined by a Bilt statement, or None for ordinary Python."""
    if isinstance(statement, (ast.FunctionDef, ast.ClassDef)):
        if any(_recognised(dec, decorators, modules, members=_DECORATORS) for dec in statement.decorator_list):
            return (statement.name,)
        return None

    value: ast.expr | None = None
    names: tuple[str, ...] = ()
    if isinstance(statement, ast.Assign):
        value = statement.value
        names = tuple(target.id for target in statement.targets if isinstance(target, ast.Name))
    elif isinstance(statement, ast.AnnAssign):
        value = statement.value
        if isinstance(statement.target, ast.Name):
            names = (statement.target.id,)
    elif isinstance(statement, ast.Expr):
        value = statement.value

    if isinstance(value, ast.Call) and _recognised(
        value.func, implementations, modules, members=frozenset({"Implementation"})
    ):
        return names
    return None


def _call(name: str, *arguments: ast.expr) -> ast.Call:
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=list(arguments), keywords=[])


def _guard(statement: ast.stmt, names: tuple[str, ...]) -> ast.Try:
    """Catch Bilt failures and missing failed dependencies for one definition."""
    declared = ast.Tuple(elts=[ast.Constant(value=name) for name in names], ctx=ast.Load())
    error = ast.Name(id=_CAUGHT, ctx=ast.Load())
    line = ast.Constant(value=statement.lineno)
    report = ast.Expr(value=_call(_REPORT, error, declared, line))
    skip_or_raise = ast.If(
        test=ast.UnaryOp(op=ast.Not(), operand=_call(_SKIP, error, declared, line)),
        body=[ast.Raise()],
        orelse=[],
    )
    return ast.copy_location(
        ast.Try(
            body=[statement],
            handlers=[
                ast.ExceptHandler(type=ast.Name(id=_ERROR, ctx=ast.Load()), name=_CAUGHT, body=[report]),
                ast.ExceptHandler(type=ast.Name(id="NameError", ctx=ast.Load()), name=_CAUGHT, body=[skip_or_raise]),
            ],
            orelse=[],
            finalbody=[],
        ),
        statement,
    )


def _instrument(source: str, filename: str) -> types.CodeType | None:
    """Rewrite only direct Bilt definitions, keeping their original source locations."""
    tree = ast.parse(source, filename=filename)
    decorators, implementations, modules = _bilt_imports(tree)
    found = False
    statements: list[ast.stmt] = []
    for statement in tree.body:
        names = _definition_names(statement, decorators, implementations, modules)
        if names is None:
            statements.append(statement)
        else:
            found = True
            statements.append(_guard(statement, names))
    if not found:
        return None
    tree.body = statements
    ast.fix_missing_locations(tree)
    return compile(tree, filename, "exec")


class _CheckingLoader(importlib.machinery.SourceFileLoader):
    """Keep standard source-loader behaviour, changing only compilation of Bilt definitions."""

    def __init__(
        self,
        fullname: str,
        filename: Path,
        root: Path,
        diagnostics: list[Diagnostic],
        skipped: list[SkippedDefinition],
    ) -> None:
        super().__init__(fullname, str(filename))
        self.filename = filename
        self.root = root
        self.diagnostics = diagnostics
        self.skipped = skipped

    def get_code(self, fullname: str) -> types.CodeType | None:
        source = self.get_source(fullname)
        if source is not None and (code := _instrument(source, str(self.filename))) is not None:
            return code
        return super().get_code(fullname)

    def exec_module(self, module: types.ModuleType) -> None:
        failed: set[str] = set()

        def mark(names: tuple[str, ...]) -> None:
            failed.update(names)

        def note(dependency: str, names: tuple[str, ...], line: int) -> None:
            mark(names)
            self.skipped.append(
                SkippedDefinition(
                    location=f"{self.filename.relative_to(self.root)}:{line}",
                    definition=", ".join(names) if names else "Implementation(...)",
                    dependency=dependency,
                )
            )

        def skip(error: NameError, names: tuple[str, ...], line: int) -> bool:
            if error.name not in failed:
                return False
            note(error.name, names, line)
            return True

        def report(error: BiltError, names: tuple[str, ...], line: int) -> None:
            if isinstance(error, BiltUnresolvedAnnotationError) and error.name in failed:
                note(error.name, names, line)
                return
            mark(names)
            for issue in error.errors if isinstance(error, BiltCollectedValidationError) else (error,):
                self.diagnostics.append(
                    Diagnostic(
                        location=f"{self.filename.relative_to(self.root)}:{line}",
                        code=code_for(issue),
                        message=f"{type(issue).__name__}: {issue}",
                    )
                )

        namespace = vars(module)
        injected = {_REPORT: report, _SKIP: skip, _ERROR: BiltError}
        original = {name: namespace[name] for name in injected if name in namespace}
        namespace.update(injected)
        try:
            super().exec_module(module)
        finally:
            for name in injected:
                namespace.pop(name, None)
            namespace.update(original)


class _ProjectFinder(importlib.abc.MetaPathFinder):
    """Wrap only ordinary Python source files inside the consumer project."""

    def __init__(self, root: Path, diagnostics: list[Diagnostic], skipped: list[SkippedDefinition]) -> None:
        self.root = root.resolve()
        self.diagnostics = diagnostics
        self.skipped = skipped

    def find_spec(
        self,
        fullname: str,
        path: collections.abc.Sequence[str] | None = None,
        target: types.ModuleType | None = None,  # noqa: ARG002 - required by finder protocol
    ) -> importlib.machinery.ModuleSpec | None:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.origin is None or type(spec.loader) is not importlib.machinery.SourceFileLoader:
            return None
        filename = Path(spec.origin).resolve()
        if filename.suffix != ".py" or not filename.is_relative_to(self.root):
            return None
        spec.loader = _CheckingLoader(fullname, filename, self.root, self.diagnostics, self.skipped)
        return spec


@contextlib.contextmanager
def recovering_project_imports(
    root: Path, diagnostics: list[Diagnostic], skipped: list[SkippedDefinition]
) -> collections.abc.Generator[None]:
    """Enable focused recovery for a check; remove the finder even on failure."""
    finder = _ProjectFinder(root, diagnostics, skipped)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
