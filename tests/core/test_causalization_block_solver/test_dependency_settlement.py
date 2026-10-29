import itertools

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._port import mangle_name
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltExecutionPlanError,
    BiltUnderconstrainedError,
)
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._system import _implementation as implementation_module
from bilt.symbolic._domains import Domain


@system
class Ports(metaclass=Interface):
    a: float = field()
    b: float = field()
    c: float = field()
    x: float = field()
    y: float = field()


@transform
def b(a: float) -> float:
    return 2 * a


@equation
def next_value(b: float, c: float) -> bool:
    return c == b + 1


@equation
def supply(source: float, child: Ports) -> bool:
    return child.a == source


@equation
def combined(answer: float, child: Ports) -> bool:
    return answer == child.b + child.c


@pytest.fixture(autouse=True)
def _causalization_children(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(implementation_module, "CausalizationStructuralAnalyzer", CausalizationStructuralAnalyzer)


def _run(analyser: CausalizationStructuralAnalyzer, inputs: dict[str, float], output: str) -> object:
    request = EvaluationRequest(output=output, input_domains=dict.fromkeys(inputs, Domain.scalar()))
    with EvaluationContext():
        case = analyser.analyse(request)
        compiled = analyser.backend.chain_case(case)
        return compiled.callable(**{mangle_name(name): value for name, value in inputs.items()})


@pytest.mark.parametrize("child_order", ((0, 1), (1, 0)))
@pytest.mark.parametrize("parent_order", tuple(itertools.permutations(range(3))))
def test_multiple_child_outputs_execute_in_every_declaration_order(
    child_order: tuple[int, ...], parent_order: tuple[int, ...]
) -> None:
    operations = (b, next_value)
    child = SubSystemConstituent("child", Implementation(Ports, [operations[index] for index in child_order]))
    constituents = (child, supply, combined)
    analyser = CausalizationStructuralAnalyzer([constituents[index] for index in parent_order], PYTHON_BACKEND)

    expected = 13.0
    assert _run(analyser, {"source": 3.0}, "answer") == expected


def test_direct_cycle_reports_unsupported_blocks() -> None:
    @transform
    def x(y: float) -> float:
        return y + 1

    @transform
    def y(x: float) -> float:
        return x / 2

    analyser = CausalizationStructuralAnalyzer([x, y], PYTHON_BACKEND)
    with pytest.raises(BiltExecutionPlanError, match="coupled blocks"):
        _run(analyser, {}, "x")


def test_structural_underconstraint_keeps_its_error_type() -> None:
    @equation
    def relation(x: float, y: float) -> bool:
        return x == y + 1

    analyser = CausalizationStructuralAnalyzer([relation], PYTHON_BACKEND)
    with pytest.raises(BiltUnderconstrainedError):
        _run(analyser, {}, "x")
