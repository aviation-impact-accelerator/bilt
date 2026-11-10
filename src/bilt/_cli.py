"""Bilt command-line interface."""

import argparse
import sys
import tomllib
from pathlib import Path

from bilt._check import check_modules
from bilt._check_discovery import discover, project_root
from bilt._errors import BiltCheckConfigurationError


def main(argv: list[str] | None = None) -> int:
    """Run a Bilt CLI command, returning its exit code."""
    parser = argparse.ArgumentParser(prog="bilt")
    commands = parser.add_subparsers(dest="command", required=True)
    checker = commands.add_parser("check", help="Check Bilt model definitions")
    checker.add_argument("path", nargs="?", type=Path, default=Path(), help="File or directory to check")
    checker.add_argument("--config", type=Path, help="Alternative pyproject.toml")
    checker.add_argument("--module", action="append", dest="modules", help="Module/package to check (repeatable)")
    args = parser.parse_args(argv)

    if args.command == "check":
        return _check(args.path, args.config, args.modules)
    return 2


def _settings(data: dict[str, object], modules_override: list[str] | None) -> tuple[list[str], bool, list[str]]:
    settings: object = data
    for section in ("tool", "bilt", "check"):
        if not isinstance(settings, dict):
            msg = "[tool.bilt.check] must be a table"
            raise BiltCheckConfigurationError(msg)
        settings = settings.get(section, {})
    if not isinstance(settings, dict):
        msg = "[tool.bilt.check] must be a table"
        raise BiltCheckConfigurationError(msg)

    modules = modules_override if modules_override is not None else settings.get("modules", [])
    recursive = settings.get("recursive", True)
    excludes = settings.get("exclude", [])
    if not isinstance(modules, list) or not all(isinstance(name, str) and name.strip() for name in modules):
        msg = "'modules' must be an array of nonempty module names"
        raise BiltCheckConfigurationError(msg)
    if not isinstance(recursive, bool):
        msg = "'recursive' must be true or false"
        raise BiltCheckConfigurationError(msg)
    if not isinstance(excludes, list) or not all(isinstance(item, str) and item.strip() for item in excludes):
        msg = "'exclude' must be an array of nonempty glob patterns"
        raise BiltCheckConfigurationError(msg)
    return modules, recursive, excludes


def _read_config(path: Path, *, required: bool) -> dict[str, object]:
    """Read optional project settings, reporting configuration failures consistently."""
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if required:
            msg = "pyproject.toml not found"
            raise BiltCheckConfigurationError(msg) from None
        return {}
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise BiltCheckConfigurationError(str(error)) from error


def _check(target: Path, config: Path | None, modules_override: list[str] | None) -> int:
    target = target.resolve()
    root = project_root(target, config)
    config_path = config.resolve() if config is not None else root / "pyproject.toml"
    try:
        data = _read_config(config_path, required=config is not None)
        modules, recursive, excludes = _settings(data, modules_override)
        found = discover(target, root, excludes)
    except BiltCheckConfigurationError as error:
        sys.stderr.write(f"{config_path}: error: {error}\n")
        return 2

    all_modules = list(dict.fromkeys((*found.modules, *modules)))
    result = check_modules(
        all_modules,
        recursive=recursive,
        recursive_roots=modules,
        project_root=root,
        import_paths=found.import_paths,
    )
    for problem in found.errors:
        sys.stderr.write(f"{problem}\n")
    for diagnostic in result.diagnostics:
        sys.stderr.write(f"{diagnostic.location}: error: {diagnostic.code} {diagnostic.message}\n")
    for skipped in result.skipped:
        sys.stderr.write(
            f"{skipped.location}: note: skipped {skipped.definition} (depends on invalid {skipped.dependency!r})\n"
        )

    sys.stdout.write(
        f"Checked {result.modules} module(s): {result.systems} system(s), "
        f"{result.implementations} implementation(s), {result.equations} equation(s), "
        f"{result.transforms} transform(s).\n"
    )
    if result.skipped:
        sys.stderr.write(f"{len(result.skipped)} definition(s) skipped.\n")
    if result.diagnostics:
        sys.stderr.write(f"{len(result.diagnostics)} errors found.\n")
    if found.errors or result.diagnostics:
        return 1
    if not result.models:
        sys.stderr.write("error: no Bilt model definitions found in the selected sources\n")
        return 1
    sys.stdout.write("No definition errors found.\n")
    return 0
