"""Find importable Bilt model definitions without importing unrelated Python files."""

import ast
import typing
from dataclasses import dataclass
from pathlib import Path

from bilt._check_api import ApiChecker, scan_api
from bilt._errors import BiltCheckConfigurationError
from bilt._source_models import DECORATORS, IMPLEMENTATIONS, bilt_imports, recognised
from bilt._source_project import source_files

if typing.TYPE_CHECKING:
    import collections.abc

@dataclass(frozen=True, slots=True)
class Discovery:
    """Candidate modules, import roots and source files that could not be inspected."""

    modules: tuple[str, ...]
    import_paths: tuple[Path, ...]
    errors: tuple[str, ...]


def _candidate(tree: ast.Module) -> bool:
    """Recognise Bilt model declarations by their direct imports."""
    decorators, implementations, modules = bilt_imports(ast.walk(tree))
    names = decorators | implementations
    members = DECORATORS | IMPLEMENTATIONS
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and any(
            recognised(decorator, names, modules, members=members) for decorator in node.decorator_list
        ):
            return True
        if isinstance(node, ast.Call) and recognised(node.func, names, modules, members=members):
            return True
    return False


def _module(file: Path, root: Path) -> tuple[str, Path] | None:
    parts = file.relative_to(root).parts
    source_index = next((index for index in reversed(range(len(parts) - 1)) if parts[index] == "src"), None)
    if source_index is not None:
        import_root = root.joinpath(*parts[: source_index + 1])
        module_parts = list(parts[source_index + 1 :])
    else:
        import_root = root
        module_parts = list(parts)
    if module_parts[-1] == "__init__.py":
        module_parts.pop()
    else:
        module_parts[-1] = file.stem
    if not module_parts or not all(part.isidentifier() for part in module_parts):
        return None
    return ".".join(module_parts), import_root


def _api_errors(sources: list[tuple[str, str, ast.Module, tuple[str, Path] | None]]) -> list[str]:
    errors: list[str] = []
    exports: dict[str, str] = {}
    # Resolve direct model imports from neighbouring source files, including re-exports.
    for _ in range(2):
        for relative, _, tree, located in sources:
            if located is None:
                continue
            module = located[0]
            package = module if relative.endswith("/__init__.py") else module.rpartition(".")[0]
            checker = ApiChecker(relative, package, exports)
            checker.visit(tree)
            exports.update({f"{module}.{name}": kind for name, kind in checker.bindings.items() if kind is not None})
    for relative, source, tree, located in sources:
        module = located[0] if located is not None else ""
        package = module if relative.endswith("/__init__.py") else module.rpartition(".")[0]
        try:
            checker = scan_api(source, tree, relative, package, exports)
        except SyntaxError as error:
            line = f":{error.lineno}" if error.lineno else ""
            errors.append(f"{relative}{line}: error: SyntaxError: {error.msg}")
            continue
        errors.extend(f"{item.location}: error: {item.code} {item.message}" for item in checker.diagnostics)
    return errors


def discover(target: Path, root: Path, excludes: collections.abc.Sequence[str]) -> Discovery:
    """Scan source safely; only candidate modules will later be imported by the checker."""
    target = target.resolve()
    root = root.resolve()
    if not target.exists() or (target.is_file() and target.suffix != ".py") or not target.is_relative_to(root):
        msg = f"{target} must be an existing Python file or directory inside {root}"
        raise BiltCheckConfigurationError(msg)

    modules: set[str] = set()
    import_paths: set[Path] = {root, root / "src"}
    errors: list[str] = []
    sources: list[tuple[str, str, ast.Module, tuple[str, Path] | None]] = []
    for file in source_files(target, root, excludes):
        relative = file.relative_to(root).as_posix()
        try:
            source = file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file))
        except (OSError, UnicodeError, SyntaxError) as error:
            line = f":{error.lineno}" if isinstance(error, SyntaxError) and error.lineno else ""
            errors.append(f"{relative}{line}: error: {type(error).__name__}: {error}")
            continue
        located = _module(file, root)
        sources.append((relative, source, tree, located))
        if not _candidate(tree):
            continue
        if located is None:
            errors.append(f"{relative}: error: cannot import this Bilt module; use an importable Python module name")
            continue
        module, import_root = located
        modules.add(module)
        import_paths.add(import_root)
    errors.extend(_api_errors(sources))
    return Discovery(tuple(sorted(modules)), tuple(sorted(import_paths)), tuple(errors))
