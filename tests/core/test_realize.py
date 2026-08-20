import ast
import textwrap

import pytest

from bilt._backend._callable import BackendCallable
from bilt._backend._python._backend import PythonBackend
from bilt.core._realize_analysis._realize import (
    NameMangler,
    RealizationFailure,
    canonicalize_backend_callable,
    mangle_name,
    realize_plan,
)
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep


def _mangled_callable(parameters: tuple[str, ...] = ("resistor__i", "resistor__r")) -> BackendCallable:
    """A compiled callable that speaks mangled parameter names."""

    def inner(*args: float) -> float:
        return sum(args)

    return BackendCallable(name="v", parameters=parameters, callable=inner)


def test_mangle_name_leaves_a_plain_name_alone() -> None:
    assert mangle_name("voltage") == "voltage"


def test_mangle_name_replaces_the_dotted_separator() -> None:
    assert mangle_name("resistor.v") == "resistor__v"


def test_mangle_name_replaces_every_level_of_a_nested_name() -> None:
    assert mangle_name("board.resistor.v") == "board__resistor__v"


def test_name_mangler_rewrites_the_signature_and_the_body() -> None:
    source = textwrap.dedent("""
        def resistor_v(resistor_i: float, resistor_r: float) -> float:
            return resistor_i * resistor_r
    """)
    function_ast = ast.parse(source).body[0]
    assert isinstance(function_ast, ast.FunctionDef)
    # Spell the dotted names in after parsing: they are not valid Python source.
    function_ast.name = "resistor.v"
    for argument in function_ast.args.args:
        argument.arg = argument.arg.replace("resistor_", "resistor.")
    for node in ast.walk(function_ast):
        if isinstance(node, ast.Name):
            node.id = node.id.replace("resistor_", "resistor.")

    mangled = NameMangler().visit(function_ast)

    assert mangled.name == "resistor__v"
    assert [argument.arg for argument in mangled.args.args] == ["resistor__i", "resistor__r"]
    assert ast.unparse(mangled.body[0]) == "return resistor__i * resistor__r"


def test_name_mangler_leaves_a_plain_function_untouched() -> None:
    source = textwrap.dedent("""
        def c(a: float, b: float) -> float:
            return a + b
    """)
    function_ast = ast.parse(source).body[0]
    assert isinstance(function_ast, ast.FunctionDef)

    mangled = NameMangler().visit(function_ast)

    assert mangled.name == "c"
    assert ast.unparse(mangled.body[0]) == "return a + b"


def test_canonicalize_relabels_the_parameters_without_touching_the_callable() -> None:
    mangled_callable = _mangled_callable()

    canonical = canonicalize_backend_callable(mangled_callable, ("resistor.i", "resistor.r"))

    assert canonical.parameters == ("resistor.i", "resistor.r")
    assert canonical.callable is mangled_callable.callable


def test_canonicalize_takes_its_parameter_order_from_the_callable() -> None:
    mangled_callable = _mangled_callable()

    canonical = canonicalize_backend_callable(mangled_callable, {"resistor.r", "resistor.i"})

    assert canonical.parameters == ("resistor.i", "resistor.r")


def test_canonicalize_rejects_a_name_the_callable_does_not_have() -> None:
    mangled_callable = _mangled_callable()

    with pytest.raises(KeyError):
        canonicalize_backend_callable(mangled_callable, ("resistor.i", "resistor.typo"))


def test_canonicalize_leaves_the_original_callable_unchanged() -> None:
    mangled_callable = _mangled_callable()

    canonicalize_backend_callable(mangled_callable, ("resistor.i", "resistor.r"))

    assert mangled_callable.parameters == ("resistor__i", "resistor__r")


def test_a_relabelled_callable_evaluates_under_its_canonical_names() -> None:
    current, resistance = 2.0, 3.0
    canonical = canonicalize_backend_callable(_mangled_callable(), ("resistor.i", "resistor.r"))

    assert canonical.evaluate({"resistor.i": current, "resistor.r": resistance}) == current + resistance


def test_realize_plan_mangles_the_case_names() -> None:
    plan = AnalysisPlan(inputs=frozenset({"resistor.i", "resistor.r"}), output="resistor.v", steps=())

    case = realize_plan(plan, PythonBackend())

    assert not isinstance(case, RealizationFailure)
    assert case.name == "resistor__v"
    assert case.return_name == "resistor__v"
    assert case.parameters == ("resistor__i", "resistor__r")


def test_realize_plan_leaves_plain_names_alone() -> None:
    plan = AnalysisPlan(inputs=frozenset({"a", "b"}), output="c", steps=())

    case = realize_plan(plan, PythonBackend())

    assert not isinstance(case, RealizationFailure)
    assert case.name == "c"
    assert case.parameters == ("a", "b")


def test_realize_plan_rejects_an_unsupported_operation() -> None:
    class Unsupported:
        __bilt_name__: str = "unsupported"
        __bilt_parameters__: tuple[str, ...] = ("a",)

    plan = AnalysisPlan(inputs=frozenset(), output="c", steps=(PlanStep(operation=Unsupported(), output="c"),))

    with pytest.raises(NotImplementedError, match="is not supported for evaluation"):
        realize_plan(plan, PythonBackend())
