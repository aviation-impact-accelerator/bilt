"""Restricted match/case syntax for symbolic equations."""

import ast
import textwrap

import pytest

from bilt.core._equation_ast import BiltEquationSyntaxError, normalise_equation_ast


def _normalise(source: str) -> ast.FunctionDef:
    node = ast.parse(textwrap.dedent(source)).body[0]
    assert isinstance(node, ast.FunctionDef)
    return normalise_equation_ast(node)


def test_match_normalises_to_nested_conditional_expressions() -> None:
    node = _normalise("""
        def relation(mode, a, b, result):
            match mode:
                case 0:
                    selected = a
                case 1:
                    selected: float = b
                case _:
                    selected = a + b
            return result == selected
    """)
    selected = node.body[0]
    assert isinstance(selected, ast.Assign)
    assert isinstance(selected.targets[0], ast.Name)
    assert selected.targets[0].id == "selected"
    assert ast.unparse(selected.value) == ("_bilt_conditional(mode == 0, a, _bilt_conditional(mode == 1, b, a + b))")
    assert not any(isinstance(child, ast.Match) for child in ast.walk(node))
    compile(ast.Module(body=[node], type_ignores=[]), "<match equation>", "exec")


def test_signed_and_float_literal_cases() -> None:
    node = _normalise("""
        def relation(mode, result):
            match mode:
                case -2:
                    selected = 10
                case 0.5:
                    selected = 20
                case _:
                    selected = 30
            return result == selected
    """)
    selected = node.body[0]
    assert isinstance(selected, ast.Assign)
    expression = ast.unparse(selected.value)
    assert "mode == -2" in expression
    assert "mode == 0.5" in expression


def test_match_and_if_can_be_nested() -> None:
    node = _normalise("""
        def relation(mode, x, result):
            if x > 0:
                match mode:
                    case 0:
                        selected = x
                    case _:
                        selected = x * 2
            else:
                selected = -x
            return result == selected
    """)
    assert not any(isinstance(child, (ast.Match, ast.If)) for child in ast.walk(node))
    compile(ast.Module(body=[node], type_ignores=[]), "<match equation>", "exec")


def test_match_can_contain_nested_if() -> None:
    node = _normalise("""
        def relation(mode, x, result):
            match mode:
                case 0:
                    if x > 0:
                        selected = x
                    else:
                        selected = -x
                case _:
                    selected = 0
            return result == selected
    """)
    assert not any(isinstance(child, (ast.Match, ast.If)) for child in ast.walk(node))
    compile(ast.Module(body=[node], type_ignores=[]), "<match equation>", "exec")


@pytest.mark.parametrize(
    ("source", "message"),
    (
        (
            "def relation(mode, y):\n    match mode:\n        case 0:\n            result = 1\n    return y == result",
            "final case _",
        ),
        (
            "def relation(mode, y):\n    match mode:\n        case 'idle':\n            result = 1\n        case _:\n            result = 0\n    return y == result",
            "numeric literals",
        ),
        (
            "def relation(mode, y):\n    match mode:\n        case True:\n            result = 1\n        case _:\n            result = 0\n    return y == result",
            "numeric literals",
        ),
        (
            "def relation(mode, y):\n    match mode:\n        case 0 | 1:\n            result = 1\n        case _:\n            result = 0\n    return y == result",
            "numeric literals",
        ),
        (
            "def relation(mode, y):\n    match mode:\n        case 0 if mode > 0:\n            result = 1\n        case _:\n            result = 0\n    return y == result",
            "match guards",
        ),
        (
            "def relation(mode, y):\n    match mode:\n        case 0:\n            result = 1\n        case other:\n            result = 0\n    return y == result",
            "final case _",
        ),
        (
            "def relation(mode, y):\n    match mode:\n        case 0:\n            result = 1\n        case _:\n            other = 0\n    return y == result",
            "same local",
        ),
        (
            "def relation(mode, y):\n    match mode:\n        case 0:\n            result = 1\n            extra = 2\n        case _:\n            result = 0\n    return y == result",
            "exactly one assignment",
        ),
        (
            "def relation(mode, y):\n    match mode:\n        case 0:\n            mode = 1\n        case _:\n            mode = 2\n    return y == mode",
            "cannot reassign equation parameter",
        ),
        (
            "def relation(mode, y):\n    match mode:\n        case 0:\n            result = later\n        case _:\n            result = 0\n    later = 2\n    return y == result",
            "before assignment",
        ),
        (
            "def relation(mode, y):\n    match later:\n        case 0:\n            result = 1\n        case _:\n            result = 2\n    later = mode\n    return y == result",
            "before assignment",
        ),
        (
            "def relation(mode, y):\n    match mode:\n        case 0:\n            result = result + 1\n        case _:\n            result = 0\n    return y == result",
            "before assignment",
        ),
        (
            "def relation(mode, y):\n    match mode:\n        case 0:\n            result = 1\n        case _:\n            result = 2\n    if mode > 0:\n        other = 1\n    else:\n        other = 2\n    return y == result + other",
            "at most one if/else",
        ),
    ),
)
def test_rejects_unsupported_match_patterns(source: str, message: str) -> None:
    with pytest.raises(BiltEquationSyntaxError, match=message):
        _normalise(source)


def test_normalisation_does_not_modify_original_match() -> None:
    source = """
        def relation(mode, y):
            match mode:
                case 0:
                    value = 1
                case _:
                    value = 2
            return y == value
    """
    original = ast.parse(textwrap.dedent(source)).body[0]
    assert isinstance(original, ast.FunctionDef)
    normalise_equation_ast(original)
    assert isinstance(original.body[0], ast.Match)
