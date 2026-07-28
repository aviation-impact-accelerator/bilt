import pytest

from bilt.core._equation import equation
from bilt.core._structural_analysis._errors import BiltSystemEvaluationError
from bilt.core._structural_analysis._exhaustive._backend import exhaustive_structural_backend
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep
from bilt.core._transform import transform


def test_unregistered_output_raises() -> None:
    with pytest.raises(BiltSystemEvaluationError, match="No operation registered"):
        exhaustive_structural_backend((), frozenset({"a"}), "z")


def test_no_valid_path_raises() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    with pytest.raises(BiltSystemEvaluationError, match="No valid execution path"):
        exhaustive_structural_backend((c,), frozenset({"a"}), "c")


def test_single_transform_all_inputs_available() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    plan = exhaustive_structural_backend((c,), frozenset({"a", "b"}), "c")

    assert plan == AnalysisPlan(
        inputs=frozenset({"a", "b"}),
        steps=(PlanStep(operation=c, output="c"),),
        output="c",
    )


def test_linear_chain_produces_correct_step_order() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    @transform
    def d(a: float, c: float) -> float:
        return a * c

    plan = exhaustive_structural_backend((c, d), frozenset({"a", "b"}), "d")

    assert plan.steps == (
        PlanStep(operation=c, output="c"),
        PlanStep(operation=d, output="d"),
    )


def test_diamond_dependency_resolves() -> None:
    @transform
    def b(a: float) -> float:
        return a + 1

    @transform
    def c(a: float) -> float:
        return a * 2

    @transform
    def d(b: float, c: float) -> float:
        return b + c

    plan = exhaustive_structural_backend((b, c, d), frozenset({"a"}), "d")

    step_outputs = [s.output for s in plan.steps]
    assert step_outputs.index("b") < step_outputs.index("d")
    assert step_outputs.index("c") < step_outputs.index("d")
    assert plan.output == "d"


def test_backtracking_selects_satisfiable_component() -> None:
    @transform
    def c(x: float) -> float:
        return x

    c_needs_x = c

    @transform
    def c(a: float) -> float:  # type: ignore[no-redef]
        return a * 2

    c_needs_a = c

    plan = exhaustive_structural_backend((c_needs_x, c_needs_a), frozenset({"a"}), "c")

    assert plan.steps == (PlanStep(operation=c_needs_a, output="c"),)


def test_equation_isolated_as_any_parameter() -> None:
    @equation
    def rel(a: float, b: float, c: float) -> bool:
        return a == b + c

    plan_c = exhaustive_structural_backend((rel,), frozenset({"a", "b"}), "c")
    assert plan_c.steps == (PlanStep(operation=rel, output="c"),)

    plan_b = exhaustive_structural_backend((rel,), frozenset({"a", "c"}), "b")
    assert plan_b.steps == (PlanStep(operation=rel, output="b"),)

    plan_a = exhaustive_structural_backend((rel,), frozenset({"b", "c"}), "a")
    assert plan_a.steps == (PlanStep(operation=rel, output="a"),)
