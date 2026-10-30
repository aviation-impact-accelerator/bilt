import dataclasses
import sys
import typing

import numpy as np
import pytest
from scipy import optimize

from bilt._backend._nonlinear_solve import NonlinearControls, NonlinearResult, NonlinearSolver
from bilt._backend._python import ScipyHybrid, ScipyLeastSquares
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.errors import BiltBackendError, BiltNonlinearSolveError


def _solve(  # noqa: PLR0913 - mirrors the execution boundary
    residual: typing.Callable,
    guess: tuple = (1.0,),
    *,
    variable_scales: tuple[float, ...] = (1.0,),
    residual_scales: tuple[float, ...] = (1.0,),
    controls: NonlinearControls | None = None,
    solver: NonlinearSolver | None = None,
) -> tuple[float, ...]:
    return PYTHON_BACKEND.solve_nonlinear(
        residual,
        guess,
        variable_scales=variable_scales,
        residual_scales=residual_scales,
        controls=NonlinearControls() if controls is None else controls,
        solver=solver,
    )


def test_scaled_coordinates_and_residuals_are_passed_to_one_numerical_method(monkeypatch: pytest.MonkeyPatch) -> None:
    original = optimize.root
    calls = 0

    def root(function: typing.Callable, guess: np.ndarray, **options: object) -> object:
        nonlocal calls
        calls += 1
        assert tuple(guess) == (1.0,)
        assert tuple(function(guess)) == (-0.5,)
        assert options["method"] == "hybr"
        assert options["jac"] is None
        return original(function, guess, **options)

    monkeypatch.setattr(optimize, "root", root)
    solution = _solve(lambda x: (x[0] ** 2 - 8,), (2.0,), variable_scales=(2.0,), residual_scales=(8.0,))
    assert solution == pytest.approx((8**0.5,))
    assert calls == 1


@pytest.mark.parametrize(
    ("option", "value", "match"),
    (
        ("initial_guess", (), "nonempty scalar vector"),
        ("initial_guess", ((1.0,),), "nonempty scalar vector"),
        ("initial_guess", ((1.0,), (1.0, 2.0)), "real scalar vector"),
        ("initial_guess", (complex(1, 2),), "finite real values"),
        ("initial_guess", (np.inf,), "finite real values"),
        ("initial_guess", (True,), "finite real values"),
        ("variable_scales", (0.0,), "must be positive"),
        ("residual_scales", (-1.0,), "must be positive"),
        ("residual_scales", (np.nan,), "finite real values"),
        ("variable_scales", (1.0, 2.0), "length 1"),
    ),
)
def test_invalid_vectors_are_rejected(option: str, value: tuple, match: str) -> None:
    arguments = {"initial_guess": (1.0,), "variable_scales": (1.0,), "residual_scales": (1.0,)}
    arguments[option] = value
    with pytest.raises(BiltNonlinearSolveError, match=match):
        PYTHON_BACKEND.solve_nonlinear(
            lambda x: (x[0],),
            arguments["initial_guess"],
            variable_scales=arguments["variable_scales"],
            residual_scales=arguments["residual_scales"],
            controls=NonlinearControls(),
        )


@pytest.mark.parametrize("value", ((np.nan,), (np.inf,), (complex(1, 1),), (1.0, 2.0), ((1.0,),)))
def test_bad_residual_values_and_non_square_vectors_are_rejected(value: tuple) -> None:
    with pytest.raises(BiltNonlinearSolveError, match="residual"):
        _solve(lambda _: value)


def test_residual_domain_failure_retains_its_cause() -> None:
    with pytest.raises(BiltNonlinearSolveError, match="residual evaluation failed") as raised:
        _solve(lambda x: (1 / x[0],), (0.0,))
    assert isinstance(raised.value.__cause__, ZeroDivisionError)


def test_hard_budget_counts_actual_residual_evaluations() -> None:
    calls = 0

    def residual(x: tuple[float, ...]) -> tuple[float, ...]:
        nonlocal calls
        calls += 1
        return (x[0] ** 2 - 2,)

    budget = 3
    with pytest.raises(BiltNonlinearSolveError, match="budget exhausted") as raised:
        _solve(residual, controls=NonlinearControls(max_evaluations=budget))
    assert calls == budget
    assert raised.value.evaluations == budget


@pytest.mark.parametrize(("success", "candidate"), ((True, 1.0), (False, 0.0)))
def test_solver_status_and_independent_scaled_residual_check_are_both_required(
    monkeypatch: pytest.MonkeyPatch, *, success: bool, candidate: float
) -> None:
    monkeypatch.setattr(
        optimize,
        "root",
        lambda *_, **__: optimize.OptimizeResult(x=np.array([candidate]), success=success, message="test"),
    )
    with pytest.raises(BiltNonlinearSolveError, match="root finding failed") as raised:
        _solve(lambda x: (x[0],), residual_scales=(0.1,))
    assert raised.value.residual_norm == abs(candidate) / 0.1
    assert raised.value.evaluations == 1


def test_failed_solve_does_not_prevent_a_later_solve() -> None:
    with pytest.raises(BiltNonlinearSolveError, match="root finding failed"):
        _solve(lambda x: (x[0] ** 2 + 1,))
    assert _solve(lambda x: (x[0] ** 2 - 4,)) == pytest.approx((2.0,))


@pytest.mark.parametrize(
    "options",
    (
        {"max_evaluations": 0},
        {"max_evaluations": True},
        {"max_evaluations": 2.5},
        {"step_tolerance": -1.0},
        {"residual_tolerance": np.inf},
        {"residual_tolerance": np.nan},
    ),
)
def test_invalid_numerical_controls_are_rejected(options: dict) -> None:
    with pytest.raises(ValueError, match="positive"):
        NonlinearControls(**options)


def test_missing_optional_dependency_is_a_capability_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    with pytest.raises(BiltBackendError, match=r"bilt\[nonlinear\]"):
        _solve(lambda x: (x[0],))


@dataclasses.dataclass
class CustomSolver:
    """An adapter outside Bilt, requiring no registration or SciPy dependency."""

    solution: object
    converged: bool = True
    name: str = "custom"
    backend: str = "python"
    calls: int = 0
    controls: NonlinearControls | None = None
    initial_value: tuple[float, ...] = ()
    initial_residual: tuple[float, ...] = ()

    def solve(
        self,
        residual: typing.Callable[[np.ndarray], np.ndarray],
        initial_guess: np.ndarray,
        *,
        controls: NonlinearControls,
    ) -> NonlinearResult[np.ndarray]:
        self.calls += 1
        self.controls = controls
        self.initial_value = tuple(initial_guess)
        self.initial_residual = tuple(residual(initial_guess))
        return NonlinearResult(np.asarray(self.solution), self.converged, "custom termination")


def test_custom_adapter_receives_scaled_problem_without_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    solver = CustomSolver((2**0.5,))
    result = _solve(
        lambda x: (x[0] ** 2 - 8,),
        (2.0,),
        variable_scales=(2.0,),
        residual_scales=(8.0,),
        solver=solver,
    )
    assert result == pytest.approx((8**0.5,))
    assert solver.calls == 1
    assert solver.initial_value == (1.0,)
    assert solver.initial_residual == (-0.5,)


@pytest.mark.parametrize(
    ("candidate", "converged", "match"),
    (
        ((1.0,), True, "root finding failed with custom"),
        ((0.0,), False, "root finding failed with custom"),
        ((np.nan,), True, "finite real values"),
        ((0.0, 0.0), True, "length 1"),
    ),
)
def test_custom_adapter_cannot_bypass_common_acceptance_checks(
    *, candidate: tuple, converged: bool, match: str
) -> None:
    with pytest.raises(BiltNonlinearSolveError, match=match):
        _solve(lambda x: (x[0],), solver=CustomSolver(candidate, converged=converged))


def test_incompatible_adapter_is_rejected_before_execution() -> None:
    solver = CustomSolver((0.0,), backend="jax")
    with pytest.raises(BiltBackendError, match="backend='python'"):
        _solve(lambda x: (x[0],), solver=solver)
    assert solver.calls == 0


def test_least_squares_minimum_is_not_silently_accepted_as_a_root() -> None:
    with pytest.raises(BiltNonlinearSolveError, match="scipy-trf") as raised:
        _solve(lambda x: (x[0] ** 2 + 1,), solver=ScipyLeastSquares())
    assert raised.value.residual_norm == pytest.approx(1)


@pytest.mark.parametrize("solver", (ScipyHybrid(), ScipyLeastSquares()), ids=("hybrid", "least-squares"))
def test_every_adapter_shares_the_hard_budget_including_jacobian_calls(solver: NonlinearSolver) -> None:
    calls = 0

    def residual(x: tuple[float, ...]) -> tuple[float, ...]:
        nonlocal calls
        calls += 1
        return (x[0] ** 2 - 2,)

    budget = 3
    with pytest.raises(BiltNonlinearSolveError, match="budget exhausted"):
        _solve(residual, solver=solver, controls=NonlinearControls(max_evaluations=budget))
    assert calls == budget
