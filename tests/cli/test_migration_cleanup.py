"""Regression tests for rule-independent migration cleanup."""

import ast
import typing
from pathlib import Path

import pytest

from bilt._migration import migrate_source
from bilt._migration_cleanup import CleanupOptions, remove_obsolete_imports

if typing.TYPE_CHECKING:
    from bilt._migration import MigrationContext

_APPROVED = frozenset({"example.old", "example.other"})


class RenameRule:
    def apply(self, node: ast.expr, context: MigrationContext) -> None:
        if isinstance(node, ast.Call) and context.resolve(node.func) == "example.old":
            context.replace(node.func, context.ensure_import("example", "new"))


def test_removes_only_newly_unused_approved_imports() -> None:
    before = "from example import old, other, keep\na = old(1)\n"
    after = "from example import old, other, keep\na = 1\n"
    assert remove_obsolete_imports(before, after, _APPROVED) == "from example import other, keep\na = 1\n"


def test_unapproved_import_is_retained() -> None:
    before = "from example import keep\na = keep(1)\n"
    after = "from example import keep\na = 1\n"
    assert remove_obsolete_imports(before, after, _APPROVED) == after


def test_multiline_import_and_alias() -> None:
    before = "from example import (\n    old as alias,\n    keep,\n)\na = alias(1)\n"
    after = "from example import (\n    old as alias,\n    keep,\n)\na = 1\n"
    assert remove_obsolete_imports(before, after, _APPROVED) == "from example import keep\na = 1\n"


def test_sole_suite_gets_pass() -> None:
    before = "if TYPE_CHECKING:\n    from example import old\na: old\n"
    after = "if TYPE_CHECKING:\n    from example import old\na: int\n"
    assert remove_obsolete_imports(before, after, _APPROVED) == "if TYPE_CHECKING:\n    pass\na: int\n"


def test_quoted_annotations_and_explicit_exports_are_retained() -> None:
    before = "from example import old, other as other\na = old(1)\nb = other(2)\nx: 'old'\n"
    after = "from example import old, other as other\na = 1\nb = 2\nx: 'old'\n"
    assert remove_obsolete_imports(before, after, _APPROVED) == after


def test_comments_and_dynamic_access_are_retained() -> None:
    for suffix in (" # important", "\n__all__ = ['old']", "\nx = globals()"):
        before = f"from example import old{suffix}\na = old(1)\n"
        after = f"from example import old{suffix}\na = 1\n"
        assert remove_obsolete_imports(before, after, _APPROVED) == after


def test_crlf_and_final_line() -> None:
    before = "from example import old\r\na = old(1)"
    after = "from example import old\r\na = 1"
    assert remove_obsolete_imports(before, after, _APPROVED) == "a = 1"


def test_engine_cleanup_is_opt_in_and_idempotent() -> None:
    source = "from example import old\nx = old(2)\n"
    rules = (RenameRule(),)
    options = CleanupOptions(removable_imports=_APPROVED)
    result = migrate_source(source, rules, cleanup=options)
    assert result.source == "from example import new\nx = new(2)\n"
    assert migrate_source(result.source, rules, cleanup=options) == result
    assert "from example import old" in migrate_source(source, rules).source
    assert "from example import old" in migrate_source(source, rules, cleanup=options, path=Path("__init__.py")).source


def test_formatter_is_optional_and_must_preserve_ast() -> None:
    options = CleanupOptions(formatter=lambda source: source.replace("x =", "x  ="))
    assert options.apply("x = 0\n", "x = 1\n") == "x  = 1\n"
    invalid = CleanupOptions(formatter=lambda source: source.replace("1", "2"))
    with pytest.raises(ValueError, match="changed the Python AST"):
        invalid.apply("x = 0\n", "x = 1\n")
