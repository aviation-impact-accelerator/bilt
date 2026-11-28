"""Shared project source selection for Bilt command-line tools."""

import os
import typing
from fnmatch import fnmatchcase
from pathlib import Path

if typing.TYPE_CHECKING:
    import collections.abc

_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
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


def source_files(target: Path, root: Path, excludes: collections.abc.Sequence[str]) -> list[Path]:
    """Select Python sources without following symlinks or excluded directories."""
    if target.is_file():
        return [target] if not target.is_symlink() else []
    paths: list[Path] = []
    for parent, directories, filenames in os.walk(target):
        here = Path(parent)
        relative = here.relative_to(root)
        directories[:] = sorted(
            directory
            for directory in directories
            if directory not in _IGNORED_DIRS
            and not directory.startswith(".")
            and not (here / directory).is_symlink()
            and not _excluded(relative / directory, excludes)
        )
        for filename in sorted(filenames):
            file = here / filename
            if file.suffix == ".py" and not file.is_symlink() and not _excluded(file.relative_to(root), excludes):
                paths.append(file)
    return paths
