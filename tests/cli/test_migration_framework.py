"""Rule-independent source migration tests."""

import ast
import typing

from bilt._migration import migrate, migrate_source

if typing.TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from bilt._migration import MigrationContext


class RenameRule:
    def apply(self, node: ast.expr, context: MigrationContext) -> None:
        if isinstance(node, ast.Call) and context.resolve(node.func) == "example.old":
            context.replace(node.func, context.ensure_import("example", "new"))


class WrapRule:
    def apply(self, node: ast.expr, context: MigrationContext) -> None:
        if isinstance(node, ast.Call) and context.resolve(node.func) == "example.wrap" and node.args:
            context.replace(node, b"(" + context.render(node.args[0]) + b")", multiline=True)


class ReviewRule:
    def apply(self, node: ast.expr, context: MigrationContext) -> None:
        if isinstance(node, ast.Call) and context.resolve(node.func) == "example.ambiguous":
            context.note(node, "ambiguous call")


def test_registered_rules_are_independent_and_compose() -> None:
    source = "from example import old, wrap\nx = wrap(old(2))\n"
    result = migrate_source(source, (RenameRule(), WrapRule()))
    assert "from example import new\n" in result.source
    assert "x = (new(2))" in result.source
    assert migrate_source(result.source, (RenameRule(), WrapRule())) == result
    assert migrate_source(source, ()).source == source


def test_scope_preserves_local_shadowing() -> None:
    source = (
        "from example import old\n"
        "def use(x):\n"
        "    value = x + 1\n"
        "    return old(value)\n"
        "def shadow(old):\n"
        "    return old(2)\n"
    )
    result = migrate_source(source, (RenameRule(),))
    assert "return new(value)" in result.source
    assert "return old(2)" in result.source


def test_comprehension_scopes_and_module_alias() -> None:
    source = "import example as ex\nx = [ex.old(v) for v in xs]\ny = [ex.old(v) for ex in xs]\n"
    result = migrate_source(source, (RenameRule(),))
    assert "x = [ex.new(v) for v in xs]" in result.source
    assert "y = [ex.old(v) for ex in xs]" in result.source


def test_import_collision_future_and_repeated_use() -> None:
    source = "from __future__ import annotations\nfrom example import old\nnew = other\na = old(1)\nb = old(2)\n"
    result = migrate_source(source, (RenameRule(),))
    assert "from __future__ import annotations\nfrom example import new as _new\n" in result.source
    assert "a = _new(1)\nb = _new(2)" in result.source


def test_utf8_comments_and_crlf() -> None:
    source = "from example import old\r\nlabel = 'é'; x = old(2)  # keep\r\n"
    result = migrate_source(source, (RenameRule(),))
    assert result.source == "from example import new\r\nfrom example import old\r\nlabel = 'é'; x = new(2)  # keep\r\n"


def test_review_notes_and_file_preview(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = "from example import old, ambiguous\na = old(1)\nb = ambiguous(2)\n"
    path = tmp_path / "model.py"
    path.write_text(source, encoding="utf-8")
    rules = (RenameRule(), ReviewRule())
    assert migrate(path, rules, write=False) == 1
    assert path.read_text(encoding="utf-8") == source
    preview = capsys.readouterr()
    assert "Would update 1 file(s)" in preview.out
    assert "3: ambiguous call" in preview.err
    assert migrate(path, rules, write=True) == 1
    assert "a = new(1)" in path.read_text(encoding="utf-8")
