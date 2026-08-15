import pytest

from bilt._backend._jax import JAXBackend
from bilt._backend._python import PythonBackend
from bilt.core._equation import equation
from bilt.core._structural_analysis._scc._backend import SCCStructuralAnalyser


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(PythonBackend(), id="python"),
        pytest.param(JAXBackend(), id="jax"),
    ],
)
def test_coupled_linear_system(backend) -> None:  # noqa: ANN001
    @equation
    def first(x: float, y: float, a: float) -> bool:
        return x + y == a

    @equation
    def second(x: float, y: float, b: float) -> bool:
        return x - y == b

    analyser = SCCStructuralAnalyser((first, second))

    case = analyser.analyse(
        available={"a", "b"},
        output="x",
        concrete_backend=backend,
    )

    callable_ = backend.chain_case(case)

    result = callable_.callable(
        a=10.0,
        b=4.0,
    )

    assert float(result) == pytest.approx(7.0)


def test_plan_for_target_returns_plan() -> None:
    @equation
    def result(
        result: float,
        value: float,
    ) -> bool:
        return result == value

    analyser = SCCStructuralAnalyser((result,))

    plan = analyser.plan_for_target(
        available={"value"},
        output="result",
    )

    assert plan is not None
    assert plan.inputs == frozenset({"value"})
    assert plan.output == "result"


def test_plan_for_target_returns_none_without_path() -> None:
    analyser = SCCStructuralAnalyser(())

    plan = analyser.plan_for_target(
        available={"value"},
        output="result",
    )

    assert plan is None
