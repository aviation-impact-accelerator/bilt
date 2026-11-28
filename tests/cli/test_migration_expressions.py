"""Regression tests for grammar-aware migration expression generation."""

import ast
import typing

import pytest

from bilt._migration import migrate_source
from bilt._migration_expressions import binary_expression, contextual_expression

if typing.TYPE_CHECKING:
    from bilt._migration import MigrationContext


@pytest.mark.parametrize(
    ("left", "operator", "right", "expected"),
    (
        ("value", "*", "meter", "value * meter"),
        ("a + b", "*", "meter", "(a + b) * meter"),
        ("value", "*", "meter / second", "value * (meter / second)"),
        ("a * b", "*", "c", "a * b * c"),
        ("a", "*", "b * c", "a * (b * c)"),
        ("a", "/", "b / c", "a / (b / c)"),
        ("a ** b", "**", "c", "(a ** b) ** c"),
        ("a", "**", "b ** c", "a ** b ** c"),
        ("-a", "**", "b", "(-a) ** b"),
        ("a", "**", "-b", "a ** -b"),
        ("a, b", "*", "c", "(a, b) * c"),
        ("a if test else b", "*", "c", "(a if test else b) * c"),
        ("lambda: a", "*", "b", "(lambda: a) * b"),
        ("a := value", "*", "b", "(a := value) * b"),
    ),
)
def test_binary_grouping(left: str, operator: str, right: str, expected: str) -> None:
    result = binary_expression(left.encode(), operator, right.encode())
    assert result.decode() == expected
    reference = ast.parse(f"(({left}) {operator} ({right}))", mode="eval")
    assert ast.dump(ast.parse(f"({result.decode()})", mode="eval")) == ast.dump(reference)


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("x = old(a, b)", "x = a * b"),
        ("x = old(a, b) + c", "x = a * b + c"),
        ("x = c * old(a, b)", "x = c * (a * b)"),
        ("x = -old(a, b)", "x = -(a * b)"),
        ("x = old(a, b) ** c", "x = (a * b) ** c"),
        ("x = old(a, b).attr", "x = (a * b).attr"),
        ("x = old(a, b)[0]", "x = (a * b)[0]"),
        ("x = fn(old(a, b))", "x = fn(a * b)"),
        ("x = [old(a, b) for a in xs]", "x = [a * b for a in xs]"),
        ("x = (old(a, b))", "x = (a * b)"),
    ),
)
def test_contextual_grouping(source: str, expected: str) -> None:
    encoded = source.encode()
    start = encoded.index(b"old(a, b)")
    end = start + len(b"old(a, b)")
    result = contextual_expression(encoded, start, end, b"a * b")
    assert (encoded[:start] + result + encoded[end:]).decode() == expected


class BinaryRule:
    def apply(self, node: ast.expr, context: MigrationContext) -> None:
        if (
            isinstance(node, ast.Call)
            and context.resolve(node.func) == "example.old"
            and len(node.args) == len(("left", "right"))
            and not node.keywords
        ):
            left, right = node.args
            context.replace_binary(node, left, "*", right)


def test_nested_edits_preserve_grouping() -> None:
    source = "from example import old\nx = old(a, old(b + c, d))\n"
    rules = (BinaryRule(),)
    result = migrate_source(source, rules)
    assert result.source == "from example import old\nx = a * ((b + c) * d)\n"
    assert not result.notes
    assert migrate_source(result.source, rules) == result


def test_utf8_source_offsets() -> None:
    source = "from example import old\nlabel = 'é'; x = old(a, b).attr\n"
    result = migrate_source(source, (BinaryRule(),))
    assert result.source == "from example import old\nlabel = 'é'; x = (a * b).attr\n"


def test_commented_expression_is_retained() -> None:
    source = "from example import old\nx = old(a, # keep\n    b)\n"
    result = migrate_source(source, (BinaryRule(),))
    assert result.source == source
    assert result.notes == ("2: review multiline or commented expression manually",)


def test_unsupported_operator() -> None:
    with pytest.raises(ValueError, match="Unsupported binary operator"):
        binary_expression(b"a", "and", b"b")
