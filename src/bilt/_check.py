"""Import model modules and report Bilt's definition-time errors."""

import contextlib
import importlib
import pkgutil
import sys
import traceback
import typing
from dataclasses import dataclass
from pathlib import Path

from bilt.core._equation import BasicEquation
from bilt.core._interface import Interface
from bilt.core._system._implementation import Implementation
from bilt.core._transform import BasicTransform

if typing.TYPE_CHECKING:
    import collections.abc
    import types


@dataclass(frozen=True, slots=True)
class Diagnostic:
    location: str
    message: str


@dataclass(frozen=True, slots=True)
class CheckResult:
    modules: int
    systems: int
    implementations: int
    equations: int
    transforms: int
    diagnostics: tuple[Diagnostic, ...]

    @property
    def models(self) -> int:
        return self.systems + self.implementations + self.equations + self.transforms


def check_modules(
    module_names: list[str],
    *,
    recursive: bool,
    project_root: Path,
    recursive_roots: collections.abc.Collection[str] | None = None,
    import_paths: collections.abc.Iterable[Path] = (),
) -> CheckResult:
    """Import configured model packages to trigger existing Bilt validation.

    Importing a module executes its Python module-level code. No dummy inputs
    are supplied, and this command does not request structural analysis or
    numerical evaluation.
    """
    seen_modules: set[str] = set()
    seen_objects: set[int] = set()
    counts = {"systems": 0, "implementations": 0, "equations": 0, "transforms": 0}
    diagnostics: list[Diagnostic] = []
    loaded = 0
    roots = set(module_names if recursive_roots is None else recursive_roots)

    def visit(name: str, *, descend: bool) -> None:
        nonlocal loaded
        if name in seen_modules:
            return
        seen_modules.add(name)
        try:
            module = importlib.import_module(name)
        except Exception as error:  # noqa: BLE001 - errors in model definitions are diagnostics
            diagnostics.append(_diagnostic(name, error, project_root))
            return

        loaded += 1
        _count_models(module, seen_objects, counts)

        if not descend or not hasattr(module, "__path__"):
            return
        try:
            children = sorted(pkgutil.iter_modules(module.__path__, f"{name}."), key=lambda item: item.name)
        except Exception as error:  # noqa: BLE001 - broken discovery must fail the check
            diagnostics.append(_diagnostic(name, error, project_root))
            return
        for child in children:
            visit(child.name, descend=True)

    with _import_paths(import_paths):
        for name in dict.fromkeys(module_names):
            visit(name, descend=recursive and name in roots)

    return CheckResult(loaded, **counts, diagnostics=tuple(diagnostics))


def _count_models(module: types.ModuleType, seen: set[int], counts: dict[str, int]) -> None:
    """Count definitions visible in the imported modules, avoiding aliases."""
    for value in vars(module).values():
        if id(value) in seen:
            continue
        if isinstance(value, Interface) and value.__module__ == module.__name__:
            kind = "systems"
        elif isinstance(value, Implementation):
            kind = "implementations"
        elif isinstance(value, BasicEquation) and value._function.__module__ == module.__name__:  # noqa: SLF001
            kind = "equations"
        elif isinstance(value, BasicTransform) and value._function.__module__ == module.__name__:  # noqa: SLF001
            kind = "transforms"
        else:
            continue
        seen.add(id(value))
        counts[kind] += 1


def _diagnostic(module: str, error: Exception, root: Path) -> Diagnostic:
    """Prefer a source location in the model repository over Bilt internals."""
    location = module
    for frame in traceback.extract_tb(error.__traceback__):
        path = Path(frame.filename).resolve()
        if path.is_relative_to(root) and not any(part in {".venv", "site-packages"} for part in path.parts):
            location = f"{path.relative_to(root)}:{frame.lineno}"
    return Diagnostic(location, f"{type(error).__name__}: {error}")


@contextlib.contextmanager
def _import_paths(paths: collections.abc.Iterable[Path]) -> collections.abc.Generator[None]:
    """Make discovered local source layouts importable for this check only."""
    inserted = [str(path) for path in paths if path.is_dir() and str(path) not in sys.path]
    sys.path[:0] = inserted
    try:
        yield
    finally:
        for path in inserted:
            sys.path.remove(path)
