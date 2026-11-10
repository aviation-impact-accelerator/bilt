"""Find importable Bilt model definitions without importing unrelated Python files."""

import ast
import os
import typing
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from bilt._errors import BiltCheckConfigurationError

if typing.TYPE_CHECKING:
    import collections.abc

_DECORATORS = frozenset({"equation", "system", "transform"})
_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "site-packages",
        "tests",
        "benchmarks",
        "scripts",
    }
)


@dataclass(frozen=True, slots=True)
class Discovery:
    """Candidate modules, import roots and source files that could not be inspected."""

    modules: tuple[str, ...]
    import_paths: tuple[Path, ...]
    errors: tuple[str, ...]


def project_root(target: Path, config: Path | None) -> Path:
    """Choose the project containing a target, preferring an explicit pyproject.toml."""
    if config is not None:
        return config.resolve().parent
    start = target if target.is_dir() else target.parent
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    if start.is_dir() and (start / "__init__.py").is_file():
        return start.parent
    return start


def _excluded(relative: Path, patterns: collections.abc.Sequence[str]) -> bool:
    path = relative.as_posix()
    return any(fnmatchcase(path, pattern) or fnmatchcase(path, pattern.rstrip("/") + "/*") for pattern in patterns)


def _sources(target: Path, root: Path, excludes: collections.abc.Sequence[str]) -> list[Path]:
    if target.is_file():
        return [target]
    paths: list[Path] = []
    for parent, directories, filenames in os.walk(target):
        here = Path(parent)
        relative = here.relative_to(root)
        directories[:] = sorted(
            directory
            for directory in directories
            if directory not in _IGNORED_DIRS
            and not directory.startswith(".")
            and not _excluded(relative / directory, excludes)
        )
        for filename in sorted(filenames):
            file = here / filename
            if file.suffix == ".py" and not _excluded(file.relative_to(root), excludes):
                paths.append(file)
    return paths


def _candidate(tree: ast.Module) -> bool:
    """Recognise Bilt's public decorators and implementation constructor by their imports."""
    names: set[str] = set()
    modules: set[str] = set()
    for statement in ast.walk(tree):
        if isinstance(statement, ast.ImportFrom) and statement.module == "bilt":
            names.update(
                alias.asname or alias.name
                for alias in statement.names
                if alias.name in _DECORATORS | {"Implementation"}
            )
        elif isinstance(statement, ast.Import):
            modules.update(alias.asname or "bilt" for alias in statement.names if alias.name == "bilt")

    def is_bilt(value: ast.expr) -> bool:
        if isinstance(value, ast.Name):
            return value.id in names
        return (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id in modules
            and value.attr in _DECORATORS | {"Implementation"}
        )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and any(
            is_bilt(decorator) for decorator in node.decorator_list
        ):
            return True
        if isinstance(node, ast.Call) and is_bilt(node.func):
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
    for file in _sources(target, root, excludes):
        relative = file.relative_to(root).as_posix()
        try:
            source = file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file))
        except (OSError, UnicodeError, SyntaxError) as error:
            line = f":{error.lineno}" if isinstance(error, SyntaxError) and error.lineno else ""
            errors.append(f"{relative}{line}: error: {type(error).__name__}: {error}")
            continue
        if not _candidate(tree):
            continue
        located = _module(file, root)
        if located is None:
            errors.append(f"{relative}: error: cannot import this Bilt module; use an importable Python module name")
            continue
        module, import_root = located
        modules.add(module)
        import_paths.add(import_root)
    return Discovery(tuple(sorted(modules)), tuple(sorted(import_paths)), tuple(errors))
