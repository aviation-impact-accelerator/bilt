import pytest

from bilt import transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver._backend import (
    CausalizationStructuralAnalyzer,
)
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltOverconstrainedError,
    BiltUnderconstrainedError,
)
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import (
    FullIncidenceGraph,
)
from bilt.core._structural_analysis._causalization_block_solver._structural_validation import (
    validate_request_structure,
)
from bilt.symbolic._domains import Domain


def test_scope_stops_at_supplied_inputs() -> None:
    @transform
    def x(a: float) -> float:
        return 2 * a

    @transform
    def unrelated(a: float, missing: float) -> float:
        return a + missing

    validate_request_structure(FullIncidenceGraph([x, unrelated]), {"a"}, "x", PYTHON_BACKEND)


def test_supplied_output_needs_no_validation() -> None:
    validate_request_structure(FullIncidenceGraph([]), {"x"}, "x", PYTHON_BACKEND)


def test_unknown_output_is_underconstrained() -> None:
    with pytest.raises(BiltUnderconstrainedError, match="x"):
        validate_request_structure(FullIncidenceGraph([]), set(), "x", PYTHON_BACKEND)


def test_missing_transform_input_is_underconstrained() -> None:
    @transform
    def x(y: float) -> float:
        return 2 * y

    with pytest.raises(BiltUnderconstrainedError, match="x"):
        validate_request_structure(FullIncidenceGraph([x]), set(), "x", PYTHON_BACKEND)


def test_transform_cannot_be_used_backwards_to_supply_an_input() -> None:
    @transform
    def y(x: float) -> float:
        return 2 * x

    with pytest.raises(BiltUnderconstrainedError, match="x"):
        validate_request_structure(FullIncidenceGraph([y]), {"y"}, "x", PYTHON_BACKEND)


def test_equivalent_competing_equations_are_rejected() -> None:
    @equation
    def first(x: float, a: float) -> bool:
        return x == 2 * a

    @equation
    def second(x: float, a: float) -> bool:
        return x == a + a

    with pytest.raises(BiltOverconstrainedError):
        validate_request_structure(FullIncidenceGraph([first, second]), {"a"}, "x", PYTHON_BACKEND)


def test_equation_and_transform_competing_for_output_are_rejected() -> None:
    @transform
    def x(a: float) -> float:
        return 2 * a

    @equation
    def other(x: float, a: float) -> bool:
        return x == a + 1

    with pytest.raises(BiltOverconstrainedError):
        validate_request_structure(FullIncidenceGraph([x, other]), {"a"}, "x", PYTHON_BACKEND)


def test_determined_target_is_allowed_with_an_underdetermined_branch() -> None:
    @equation
    def target(x: float, a: float) -> bool:
        return x == a

    @equation
    def branch(y: float, x: float, z: float) -> bool:
        return y == x + z

    validate_request_structure(FullIncidenceGraph([target, branch]), {"a"}, "x", PYTHON_BACKEND)


def test_analyser_rejects_overconstraint_before_trying_isolation() -> None:
    @equation
    def square(z: float, x: float) -> bool:
        return z == x * x

    @equation
    def linear(x: float, w: float) -> bool:
        return x == w + 1

    analyser = CausalizationStructuralAnalyzer([square, linear], PYTHON_BACKEND)
    request = EvaluationRequest(output="x", input_domains=dict.fromkeys({"z", "w"}, Domain.scalar()))

    with pytest.raises(BiltOverconstrainedError):
        analyser.analyse(request)
