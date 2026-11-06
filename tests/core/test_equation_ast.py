"""Contract tests for equation AST validation and normalisation."""

import ast
import textwrap

import pytest

from bilt.core._equation_ast import (
    BiltEquationSyntaxError,
    extract_equation_ast,
    normalise_equation_ast,
)

EXPECTED_SIMPLE_BODY_LENGTH = 2
EXPECTED_EXTENDED_BODY_LENGTH = 3
CONDITIONAL_ARITY = 3


def _normalise(source: str) -> ast.FunctionDef:
    node = ast.parse(textwrap.dedent(source)).body[0]
    assert isinstance(node, ast.FunctionDef)
    return normalise_equation_ast(node)


def test_simple_if_else_normalises_to_a_conditional_call() -> None:
    node = _normalise("""
        def equation(x, y):
            if x > 0:
                value = x**2
            else:
                value = -x
            return y == value
    """)
    assert len(node.body) == EXPECTED_SIMPLE_BODY_LENGTH
    assignment = node.body[0]
    assert isinstance(assignment, ast.Assign)
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == "_bilt_conditional"
    assert len(assignment.value.args) == CONDITIONAL_ARITY
    assert not any(isinstance(item, ast.If) for item in ast.walk(node))
    assert isinstance(node.body[-1], ast.Return)
    assert all(hasattr(item, "lineno") for item in ast.walk(node) if isinstance(item, ast.stmt))


def test_docstring_is_allowed_before_an_equation_return() -> None:
    node = _normalise('''
        def equation(x, y):
            """A mathematical relation."""
            return y == x + 1
    ''')
    assert ast.get_docstring(node) == "A mathematical relation."


def test_inline_conditional_is_rejected_until_it_can_be_normalised() -> None:
    with pytest.raises(BiltEquationSyntaxError, match="IfExp"):
        _normalise("""
            def equation(x, y):
                return y == (x if x > 0 else -x)
        """)


def test_plain_equation_ast_is_unchanged() -> None:
    node = _normalise("""
        def equation(x, y):
            result = x + 1
            return y == result
    """)
    assert isinstance(node.body[0], ast.Assign)
    assert not any(isinstance(item, ast.Call) for item in ast.walk(node))


@pytest.mark.parametrize(
    ("source", "message"),
    (
        ("def eq(x):\n    return x == 1\n    return x == 2", "exactly one return"),
        ("def eq(x):\n    return x, x", "tuple or list"),
        ("def eq(x):\n    return [x == 1]", "tuple or list"),
        ("def eq(x):\n    for i in range(2):\n        x = i\n    return x == 1", "For"),
        ("def eq(x):\n    x.value = 2\n    return x == 2", "assignment target"),
        ("def eq(x):\n    x[0] = 2\n    return x == 2", "assignment target"),
        ("def eq(x):\n    if x > 0:\n        a = x\n    else:\n        b = x\n    return a == b", "same local"),
        ("def eq(x):\n    if x > 0:\n        a = x\n    return a == x", "exactly one assignment"),
        (
            "def eq(x):\n    if x > 0:\n        a = x\n        b = x\n    else:\n        a = x\n    return a == x",
            "exactly one assignment",
        ),
        (
            "def eq(x):\n    if x > 0:\n        if x > 1:\n            a = x\n    else:\n        a = x\n    return a == x",
            "simple assignment",
        ),
        ("def eq(x):\n    value = missing\n    missing = x\n    return value == x", "before assignment"),
        ("def eq(x):\n    value = x\n    return value == (x := 1)", "NamedExpr"),
        ("def eq(x):\n    value = [v for v in [x]]\n    return value == x", "ListComp"),
    ),
)
def test_rejects_unsupported_syntax(source: str, message: str) -> None:
    with pytest.raises(BiltEquationSyntaxError, match=message):
        _normalise(source)


def test_extract_source_removes_decorator_without_changing_function() -> None:
    def identity[T](function: T) -> T:
        return function

    @identity
    def relation(x: float, y: float) -> bool:
        return x == y

    node = extract_equation_ast(relation)
    assert node.name == "relation"
    assert node.decorator_list == []
    assert relation(1.0, 1.0)
    assert not relation(1.0, 2.0)


def test_annotated_local_assignment_is_preserved() -> None:
    node = _normalise("""
        def equation(x, y):
            result: float = x ** 2
            return y == result
    """)
    assignment = node.body[0]
    assert isinstance(assignment, ast.AnnAssign)
    assert isinstance(assignment.target, ast.Name)
    assert assignment.target.id == "result"
    assert ast.unparse(assignment.annotation) == "float"


@pytest.mark.parametrize(
    "source",
    (
        """
        def equation(x, y):
            if x > 0:
                value: float = x**2
            else:
                value: float = -x
            return y == value
        """,
        """
        def equation(x, y):
            if x > 0:
                value: float = x**2
            else:
                value = -x
            return y == value
        """,
    ),
)
def test_annotated_conditional_assignment_normalises(source: str) -> None:
    node = _normalise(source)
    assignment = node.body[0]
    assert isinstance(assignment, ast.Assign)
    assert isinstance(assignment.targets[0], ast.Name)
    assert assignment.targets[0].id == "value"
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == "_bilt_conditional"
    assert len(assignment.value.args) == CONDITIONAL_ARITY
    compile(ast.Module(body=[node], type_ignores=[]), "<normalised equation>", "exec")


def test_annotated_local_is_checked_for_use_before_assignment() -> None:
    with pytest.raises(BiltEquationSyntaxError, match="before assignment"):
        _normalise("""
            def equation(x, y):
                result = value + x
                value: float = x**2
                return y == result
        """)


def test_annotated_declaration_without_value_is_rejected() -> None:
    with pytest.raises(BiltEquationSyntaxError, match="expected one simple assignment"):
        _normalise("""
            def equation(x, y):
                value: float
                return y == value
        """)


def test_annotated_attribute_assignment_is_rejected() -> None:
    with pytest.raises(BiltEquationSyntaxError, match="assignment target must be a local name"):
        _normalise("""
            def equation(x, y):
                x.value: float = 2
                return y == x
        """)


def test_augmented_assignment_is_still_deferred() -> None:
    with pytest.raises(BiltEquationSyntaxError, match="AugAssign"):
        _normalise("""
            def equation(x, y):
                value = x**2
                value += x
                return y == value
        """)


@pytest.mark.parametrize(
    ("source", "message"),
    (
        ("def eq(x, y):\n    value = x**2\n    value = value + x\n    return y == value", "assigned more than once"),
        ("def eq(x, y):\n    value: float = x**2\n    value = x\n    return y == value", "assigned more than once"),
        ("def eq(x, y):\n    value = x\n    value: float = x**2\n    return y == value", "assigned more than once"),
        ("def eq(x, y):\n    x = y + 1\n    return y == x", "cannot reassign equation parameter"),
        ("def eq(x, y):\n    x: float = y + 1\n    return y == x", "cannot reassign equation parameter"),
        (
            "def eq(x, y):\n    value = x\n    if x > 0:\n        value = x**2\n    else:\n        value = -x\n    return y == value",
            "assigned more than once",
        ),
        (
            "def eq(x, y):\n    if x > 0:\n        value = x**2\n    else:\n        value = -x\n    value = value + 1\n    return y == value",
            "assigned more than once",
        ),
        (
            "def eq(x, y):\n    if x > 0:\n        x = 1\n    else:\n        x = 2\n    return y == x",
            "cannot reassign equation parameter",
        ),
    ),
)
def test_rejects_rebinding(source: str, message: str) -> None:
    with pytest.raises(BiltEquationSyntaxError, match=message):
        _normalise(source)


def test_distinct_local_bindings_are_accepted() -> None:
    node = _normalise("""
        def eq(x, y):
            squared: float = x**2
            result = squared + x
            return y == result
    """)
    assert len(node.body) == EXPECTED_EXTENDED_BODY_LENGTH


def test_conditional_merge_counts_as_one_local_binding() -> None:
    node = _normalise("""
        def eq(x, y):
            squared = x**2
            if x > 0:
                result: float = squared
            else:
                result = -squared
            return y == result
    """)
    assert isinstance(node.body[1], ast.Assign)
    assert isinstance(node.body[1].value, ast.Call)
    compile(ast.Module(body=[node], type_ignores=[]), "<normalised equation>", "exec")


def test_normalisation_does_not_modify_input_ast() -> None:
    source = """
        def eq(x, y):
            if x > 0:
                result = x**2
            else:
                result = -x
            return y == result
    """
    original = ast.parse(textwrap.dedent(source)).body[0]
    assert isinstance(original, ast.FunctionDef)
    normalise_equation_ast(original)
    assert isinstance(original.body[0], ast.If)
