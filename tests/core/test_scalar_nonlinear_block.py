import typing

import numpy as np
import pytest

from bilt import Tensor, dim, domain, equation
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python import ScipyHybrid, ScipyLeastSquares
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._structural_analysis._plan import PlanBlock, PlanStep
from bilt.core._structural_analysis._prepare_nonlinear import PreparedNonlinearBlock, prepare_nonlinear_block
from bilt.core._structural_analysis._qualified_block import QualifiedBlock, QualifiedEquation
from bilt.core._structural_analysis._residual_block import extract_residual_block
from bilt.errors import BiltInputDimensionalityError, BiltNonlinearBlockError, BiltNonlinearSolveError
from bilt.units import Quantity
from bilt.units.dimensions import ABSOLUTE, LENGTH, TEMPERATURE
from bilt.units.units import absolute_celsius, absolute_kelvin, centimeter, kelvin, meter

Length = typing.Annotated[Quantity, LENGTH]
Area = typing.Annotated[Quantity, LENGTH**2]


@equation
def area_relation(width: Length, height: Length, area: Area) -> bool:
    return width * height == area


@equation
def perimeter_relation(width: Length, height: Length, perimeter: Length) -> bool:
    return 2 * (width + height) == perimeter


def _rectangle() -> PreparedNonlinearBlock:
    block = PlanBlock((PlanStep(area_relation, "width"), PlanStep(perimeter_relation, "height")), is_cyclic=True)
    return prepare_nonlinear_block(extract_residual_block(block), PYTHON_BACKEND)


def _arguments() -> dict:
    return {
        "inputs": {"area": Quantity(60000, centimeter**2), "perimeter": Quantity(1000, centimeter)},
        "initial_guess": {"width": Quantity(100, centimeter), "height": Quantity(4, meter)},
        "variable_scales": {"width": Quantity(1, meter), "height": Quantity(100, centimeter)},
        # Local row order follows unknowns: height/perimeter then width/area.
        "residual_scales": (Quantity(10, meter), Quantity(6, meter**2)),
    }


def test_rectangle_preserves_mixed_residual_units_and_lifts_each_solution() -> None:
    prepared = _rectangle()
    assert prepared.block.unknowns == ("height", "width")
    assert prepared.residual_declarations["0"].dimensionality == LENGTH
    assert prepared.residual_declarations["1"].dimensionality == LENGTH**2

    result = prepared.solve(**_arguments())

    assert isinstance(result["width"], Quantity)
    assert isinstance(result["height"], Quantity)
    assert result["width"].unit is meter
    assert result["height"].unit is meter
    assert result["width"].value == pytest.approx(2)
    assert result["height"].value == pytest.approx(3)
    assert result["width"].value * result["height"].value == pytest.approx(6)
    assert 2 * (result["width"].value + result["height"].value) == pytest.approx(10)


def test_reusing_preparation_with_different_guesses_selects_the_other_root() -> None:
    prepared = _rectangle()
    arguments = _arguments()
    first = prepared.solve(**arguments)
    arguments["initial_guess"] = {"width": Quantity(4, meter), "height": Quantity(1, meter)}
    second = prepared.solve(**arguments)
    assert isinstance(first["width"], Quantity)
    assert isinstance(second["width"], Quantity)
    assert isinstance(second["height"], Quantity)
    assert first["width"].value == pytest.approx(2)
    assert second["width"].value == pytest.approx(3)
    assert second["height"].value == pytest.approx(2)


def test_failed_execution_does_not_invalidate_prepared_block() -> None:
    prepared = _rectangle()
    arguments = _arguments()
    # No real rectangle with positive area and zero perimeter.
    arguments["inputs"] = {"area": Quantity(6, meter**2), "perimeter": Quantity(0, meter)}
    with pytest.raises(BiltNonlinearSolveError):
        prepared.solve(**arguments)
    result = prepared.solve(**_arguments())["width"]
    assert isinstance(result, Quantity)
    assert result.value == pytest.approx(2)


@pytest.mark.parametrize("argument", ("inputs", "initial_guess", "variable_scales"))
def test_guesses_and_scales_require_exact_names_separate_from_inputs(argument: str) -> None:
    arguments = _arguments()
    arguments[argument] = {**arguments[argument], "extra": 1.0}
    with pytest.raises(BiltNonlinearBlockError, match="Expected"):
        _rectangle().solve(**arguments)


@pytest.mark.parametrize("argument", ("initial_guess", "variable_scales"))
def test_wrong_unknown_units_are_rejected(argument: str) -> None:
    arguments = _arguments()
    arguments[argument]["width"] = Quantity(1, meter**2)
    with pytest.raises(BiltInputDimensionalityError):
        _rectangle().solve(**arguments)


def test_residual_scale_units_follow_equations_rather_than_unknowns() -> None:
    arguments = _arguments()
    arguments["residual_scales"] = (Quantity(1, meter), Quantity(1, meter))
    with pytest.raises(BiltInputDimensionalityError):
        _rectangle().solve(**arguments)


def test_each_residual_requires_an_explicit_scale() -> None:
    arguments = _arguments()
    arguments["residual_scales"] = (Quantity(1, meter),)
    with pytest.raises(BiltNonlinearBlockError, match="every equation row"):
        _rectangle().solve(**arguments)


def test_backend_capability_and_rectangular_blocks_fail_during_preparation() -> None:
    block = QualifiedBlock((QualifiedEquation(area_relation),), ("width", "height"))
    residual = extract_residual_block(block)
    with pytest.raises(BiltNonlinearBlockError, match="backend does not support"):
        prepare_nonlinear_block(residual, JAX_BACKEND)
    with pytest.raises(BiltNonlinearBlockError, match="equally many"):
        prepare_nonlinear_block(residual, PYTHON_BACKEND)


@pytest.mark.parametrize("bad_value", (np.nan, np.inf, complex(1, 1), [1.0], np.array([1.0])))
def test_plain_inputs_must_be_finite_real_scalars(bad_value: object) -> None:
    @equation
    def relation(x: float, p: float) -> bool:
        return x**2 == p

    prepared = prepare_nonlinear_block(extract_residual_block(PlanBlock((PlanStep(relation, "x"),))), PYTHON_BACKEND)
    with pytest.raises(BiltNonlinearBlockError, match="finite real scalar"):
        prepared.solve({"p": bad_value}, initial_guess={"x": 1}, variable_scales={"x": 1}, residual_scales=(1,))
    assert prepared.solve({"p": 4}, initial_guess={"x": 1}, variable_scales={"x": 1}, residual_scales=(1,))[
        "x"
    ] == pytest.approx(2)


def test_non_scalar_domain_hint_is_rejected() -> None:
    time = dim("time")
    series = typing.Annotated[Tensor, domain(time)]

    @equation
    def relation(x: series, p: series) -> bool:
        return x * x == p

    residual = extract_residual_block(PlanBlock((PlanStep(relation, "x"),)))
    with pytest.raises(BiltNonlinearBlockError, match="non-scalar domain"):
        prepare_nonlinear_block(residual, PYTHON_BACKEND)


def test_quantity_constants_are_compiled_in_coherent_units() -> None:
    @equation
    def relation(x: Length) -> bool:
        return x * x == Quantity(40000, centimeter**2)

    prepared = prepare_nonlinear_block(extract_residual_block(PlanBlock((PlanStep(relation, "x"),))), PYTHON_BACKEND)
    result = prepared.solve(
        {},
        initial_guess={"x": Quantity(1, meter)},
        variable_scales={"x": Quantity(1, meter)},
        residual_scales=(Quantity(1, meter**2),),
    )["x"]
    assert isinstance(result, Quantity)
    assert result.unit is meter
    assert result.value == pytest.approx(2)


def test_absolute_guesses_and_results_use_relative_scales() -> None:
    absolute_temperature = typing.Annotated[Quantity, TEMPERATURE, ABSOLUTE]

    @equation
    def relation(x: absolute_temperature, target: absolute_temperature) -> bool:
        return x == target

    prepared = prepare_nonlinear_block(extract_residual_block(PlanBlock((PlanStep(relation, "x"),))), PYTHON_BACKEND)
    arguments: dict = {
        "inputs": {"target": Quantity(100, absolute_celsius)},
        "initial_guess": {"x": Quantity(20, absolute_celsius)},
        "variable_scales": {"x": Quantity(10, kelvin)},
        "residual_scales": (Quantity(1, kelvin),),
    }
    result = prepared.solve(**arguments)["x"]
    assert isinstance(result, Quantity)
    assert result.unit is absolute_kelvin
    assert result.value == pytest.approx(373.15)
    arguments["variable_scales"]["x"] = Quantity(10, absolute_kelvin)
    with pytest.raises(BiltInputDimensionalityError):
        prepared.solve(**arguments)


def test_conflicting_quantity_declarations_fail_before_compilation() -> None:
    @equation
    def plain(width: float, height: float, p: float) -> bool:
        return width * height == p

    block = QualifiedBlock((QualifiedEquation(plain), QualifiedEquation(area_relation)), ("width", "height"))
    with pytest.raises(BiltNonlinearBlockError, match="Conflicting"):
        prepare_nonlinear_block(extract_residual_block(block), PYTHON_BACKEND)


def test_algorithm_can_change_without_recompiling_or_repreparing(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = _rectangle()
    compiled = prepared.residual

    def unexpected_compile(*_: object, **__: object) -> typing.NoReturn:
        pytest.fail("Changing algorithms must not recompile residuals")

    monkeypatch.setattr(PYTHON_BACKEND, "transpile", unexpected_compile)
    for solver in (ScipyHybrid(), ScipyLeastSquares()):
        result = prepared.solve(**_arguments(), solver=solver)
        assert isinstance(result["width"], Quantity)
        assert isinstance(result["height"], Quantity)
        assert result["width"].value == pytest.approx(2)
        assert result["height"].value == pytest.approx(3)
        assert result["width"].unit is meter
        assert result["height"].unit is meter
        assert prepared.residual is compiled
