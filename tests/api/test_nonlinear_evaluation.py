import dataclasses
import re
import runpy
import typing
from pathlib import Path

import numpy as np
import pytest

from bilt import (
    JAX_BACKEND,
    PYTHON_BACKEND,
    Implementation,
    Interface,
    Tensor,
    dim,
    equation,
    field,
    system,
    tensor,
    transform,
)
from bilt.errors import BiltAnalysisError, BiltNonlinearBlockError, BiltNonlinearSolveError
from bilt.numerics import (
    EvaluationOptions,
    NonlinearResult,
    NonlinearSolver,
    RootControls,
    RootOptions,
    ScipyHybrid,
    ScipyLeastSquares,
    VariableOptions,
)
from bilt.units import Quantity
from bilt.units.dimensions import LENGTH
from bilt.units.units import centimeter, meter

Length = typing.Annotated[Quantity, LENGTH]
Area = typing.Annotated[Quantity, LENGTH**2]


@system
class Rectangle(metaclass=Interface):
    width: Length = field()
    height: Length = field()
    area: Area = field()
    perimeter: Length = field()
    diagonal_squared: Area = field()


@equation
def area_relation(width: Length, height: Length, area: Area) -> bool:
    return width * height == area


@equation
def perimeter_relation(width: Length, height: Length, perimeter: Length) -> bool:
    return 2 * (width + height) == perimeter


@transform
def diagonal_squared(width: Length, height: Length) -> Area:
    return width * width + height * height


def _model() -> Implementation:
    return Implementation(Rectangle, [area_relation, perimeter_relation, diagonal_squared])


def _options(*, width: float = 1, height: float = 4) -> EvaluationOptions:
    return EvaluationOptions(
        nonlinear=RootOptions(
            variables={
                "width": VariableOptions(initial=Quantity(width, meter), scale=Quantity(100, centimeter)),
                "height": VariableOptions(initial=Quantity(height, meter), scale=Quantity(1, meter)),
            },
            residual_scales={area_relation: Quantity(6, meter**2), perimeter_relation: Quantity(10, meter)},
        )
    )


def _rectangle() -> Rectangle:
    return Rectangle(area=Quantity(60000, centimeter**2), perimeter=Quantity(1000, centimeter))


@pytest.mark.parametrize("solver", (ScipyHybrid(), ScipyLeastSquares()), ids=("hybrid", "least-squares"))
def test_normal_attribute_access_solves_a_mixed_unit_rectangle(solver: NonlinearSolver) -> None:
    options = _options()
    assert options.nonlinear is not None
    options = dataclasses.replace(options, nonlinear=dataclasses.replace(options.nonlinear, solver=solver))
    with _model().with_options(options):
        rectangle = _rectangle()
        assert rectangle.width.convert_to(meter).value == pytest.approx(2)
        assert rectangle.height.convert_to(meter).value == pytest.approx(3)
        assert rectangle.diagonal_squared.convert_to(meter**2).value == pytest.approx(13)


def test_nested_options_choose_different_roots_on_the_same_instance_and_restore_cache() -> None:
    rectangle = _rectangle()
    first = _options()
    second = _options(width=4, height=1)
    model = _model()
    with model:
        with model.with_options(first):
            original = rectangle.width
            assert original.value == pytest.approx(2)
            with model.with_options(second):
                assert rectangle.width.value == pytest.approx(3)
            assert rectangle.width is original
        # Guesses and calculated answers never become boundary inputs/default answers.
        with pytest.raises(BiltAnalysisError):
            _ = rectangle.width


def test_numerical_failure_does_not_blacklist_a_block_or_cache_an_answer() -> None:
    model = _model()
    rectangle = _rectangle()
    options = _options()
    assert options.nonlinear is not None
    failed = dataclasses.replace(
        options,
        nonlinear=dataclasses.replace(options.nonlinear, controls=RootControls(max_evaluations=1)),
    )
    with model:
        with model.with_options(failed), pytest.raises(BiltNonlinearSolveError, match="budget"):
            _ = rectangle.width
        with model.with_options(options):
            assert rectangle.width.value == pytest.approx(2)
        bad_inputs = Rectangle(area=Quantity(6, meter**2), perimeter=Quantity(0, meter))
        with model.with_options(options), pytest.raises(BiltNonlinearSolveError):
            _ = bad_inputs.width
        with model.with_options(_options(width=4, height=1)):
            assert rectangle.width.value == pytest.approx(3)


def test_supplied_values_remain_fixed_and_need_no_solver_configuration() -> None:
    with _model().with_options(_options()):
        supplied = Quantity(7, meter)
        rectangle = Rectangle(width=supplied)
        assert rectangle.width is supplied


def test_missing_configuration_uses_the_analysis_failure_path_then_can_recover() -> None:
    model = _model()
    rectangle = _rectangle()
    options = _options()
    assert options.nonlinear is not None
    incomplete = dataclasses.replace(options, nonlinear=dataclasses.replace(options.nonlinear, residual_scales={}))
    with model:
        with model.with_options(incomplete), pytest.raises(BiltAnalysisError):
            _ = rectangle.width
        with model.with_options(options):
            assert rectangle.width.value == pytest.approx(2)


def test_numerical_changes_reuse_compilation_but_not_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    original = PYTHON_BACKEND.transpile
    residual_compilations = 0

    def counted(definition: typing.Any) -> typing.Any:  # noqa: ANN401
        nonlocal residual_compilations
        if definition.name.startswith("__bilt_nonlinear_residual"):
            residual_compilations += 1
        return original(definition)

    monkeypatch.setattr(PYTHON_BACKEND, "transpile", counted)
    model = _model()
    rectangle = _rectangle()
    first = _options()
    second = _options(width=4, height=1)
    assert second.nonlinear is not None
    second = dataclasses.replace(second, nonlinear=dataclasses.replace(second.nonlinear, solver=ScipyLeastSquares()))
    with model:
        with model.with_options(first):
            assert rectangle.width.value == pytest.approx(2)
        with model.with_options(second):
            assert rectangle.width.value == pytest.approx(3)
        with model.with_options(first):
            other = Rectangle(area=Quantity(12, meter**2), perimeter=Quantity(14, meter))
            assert other.width.value == pytest.approx(3)
    assert residual_compilations == 1


def test_one_numerical_solve_supplies_both_unknowns_to_a_downstream_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    original = PYTHON_BACKEND.solve_nonlinear
    calls = 0

    def counted(*args: typing.Any, **kwargs: typing.Any) -> tuple[float, ...]:  # noqa: ANN401
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(PYTHON_BACKEND, "solve_nonlinear", counted)
    with _model().with_options(_options()):
        rectangle = _rectangle()
        first = rectangle.diagonal_squared
        assert first.value == pytest.approx(13)
        assert rectangle.diagonal_squared is first
    assert calls == 1


def test_cached_python_answer_is_not_returned_for_an_unsupported_backend() -> None:
    model = _model()
    rectangle = _rectangle()
    options = _options()
    with model.with_options(options):
        assert rectangle.width.value == pytest.approx(2)
        with JAX_BACKEND, pytest.raises(BiltAnalysisError, match="backend does not support"):
            _ = rectangle.width
        assert rectangle.width.value == pytest.approx(2)


def test_configuration_dictionaries_are_snapshots() -> None:
    base = _options()
    assert base.nonlinear is not None
    variables = dict(base.nonlinear.variables)
    residuals = dict(base.nonlinear.residual_scales)
    options = EvaluationOptions(nonlinear=RootOptions(variables=variables, residual_scales=residuals))
    variables["width"] = VariableOptions(initial=Quantity(4, meter), scale=Quantity(1, meter))
    residuals.clear()
    with _model().with_options(options):
        assert _rectangle().width.value == pytest.approx(2)


def test_a_previous_isolation_failure_does_not_exclude_a_scalar_nonlinear_route() -> None:
    @system
    class Scalar(metaclass=Interface):
        x: float = field()
        p: float = field()

    @equation
    def relation(x: float, p: float) -> bool:
        return x * x + x == p

    model = Implementation(Scalar, [relation])
    options = EvaluationOptions(
        nonlinear=RootOptions(variables={"x": VariableOptions(initial=1, scale=1)}, residual_scales={relation: 1})
    )
    sample = Scalar(p=6)
    with model:
        with pytest.raises(BiltAnalysisError):
            _ = sample.x
        with model.with_options(options):
            assert sample.x == pytest.approx(2)


def test_linear_blocks_take_precedence_without_requiring_guesses(monkeypatch: pytest.MonkeyPatch) -> None:
    @system
    class Pair(metaclass=Interface):
        x: float = field()
        y: float = field()
        p: float = field()
        q: float = field()

    @equation
    def total(x: float, y: float, p: float) -> bool:
        return x + y == p

    @equation
    def difference(x: float, y: float, q: float) -> bool:
        return x - y == q

    def unexpected(*_: object, **__: object) -> typing.NoReturn:
        pytest.fail("A supported linear block must not invoke the nonlinear solver")

    monkeypatch.setattr(PYTHON_BACKEND, "solve_nonlinear", unexpected)
    options = EvaluationOptions(nonlinear=RootOptions(variables={}, residual_scales={}))
    with Implementation(Pair, [total, difference]).with_options(options):
        assert Pair(p=10, q=2).x == pytest.approx(6)


def test_exception_exit_restores_the_outer_configuration() -> None:
    rectangle = _rectangle()
    options = _options()
    assert options.nonlinear is not None
    failed = dataclasses.replace(options, nonlinear=dataclasses.replace(options.nonlinear, residual_scales={}))
    model = _model()
    with model.with_options(options):
        original = rectangle.width
        with pytest.raises(BiltAnalysisError, match="Missing residual scales"), model.with_options(failed):
            _ = rectangle.width
        assert rectangle.width is original


def test_unknown_variable_names_are_rejected_instead_of_ignored() -> None:
    options = _options()
    assert options.nonlinear is not None
    typo = dataclasses.replace(
        options,
        nonlinear=dataclasses.replace(
            options.nonlinear, variables={**options.nonlinear.variables, "windth": VariableOptions(initial=1, scale=1)}
        ),
    )
    with _model().with_options(typo), pytest.raises(BiltNonlinearBlockError, match="windth"):
        _ = _rectangle().width


def test_independent_scalar_blocks_use_their_own_settings_and_runtime_inputs() -> None:
    @system
    class Pair(metaclass=Interface):
        x: float = field()
        y: float = field()
        p: float = field()
        q: float = field()
        total: float = field()

    @equation
    def first(x: float, p: float) -> bool:
        return x * x + x == p

    @equation
    def second(y: float, q: float) -> bool:
        return y * y + y == q

    @transform
    def total(x: float, y: float) -> float:
        return x + y

    options = EvaluationOptions(
        nonlinear=RootOptions(
            variables={"x": VariableOptions(initial=1, scale=1), "y": VariableOptions(initial=1, scale=1)},
            residual_scales={first: 1, second: 1},
        )
    )
    with Implementation(Pair, [first, second, total]).with_options(options):
        assert Pair(p=6, q=12).total == pytest.approx(5)
        with pytest.raises(BiltNonlinearSolveError):
            _ = Pair(p=6, q=-1).total
        assert Pair(p=12, q=20).total == pytest.approx(7)


def test_linear_fallback_preserves_tensor_domains() -> None:
    @system
    class Scalar(metaclass=Interface):
        x: Tensor = field()
        p: Tensor = field()

    @equation
    def repeated(x: float, p: float) -> bool:
        return x + x == p

    time = dim("time")
    options = EvaluationOptions(nonlinear=RootOptions(variables={}, residual_scales={}))
    with Implementation(Scalar, [repeated]).with_options(options):
        result = Scalar(p=tensor(np.array([2.0, 4.0]), over=(time,))).x
        assert isinstance(result, Tensor)
        assert result.domain.dims == (time,)
        np.testing.assert_allclose(result.values, [1.0, 2.0])


def test_transform_cycles_are_not_executed_by_the_nonlinear_solver() -> None:
    @system
    class Pair(metaclass=Interface):
        x: float = field()
        y: float = field()
        p: float = field()

    @transform
    def x(y: float) -> float:
        pytest.fail(f"A transform cycle must not be executed as residuals: {y}")

    @equation
    def relation(x: float, y: float, p: float) -> bool:
        return x * y == p

    options = EvaluationOptions(
        nonlinear=RootOptions(
            variables={"x": VariableOptions(initial=1, scale=1), "y": VariableOptions(initial=1, scale=1)},
            residual_scales={relation: 1},
        )
    )
    with Implementation(Pair, [x, relation]).with_options(options), pytest.raises(BiltAnalysisError):
        _ = Pair(p=2).x


def test_array_inputs_do_not_silently_enable_nonlinear_tensor_solves() -> None:
    @system
    class Scalar(metaclass=Interface):
        x: Tensor = field()
        p: Tensor = field()

    @equation
    def relation(x: float, p: float) -> bool:
        return x * x + x == p

    time = dim("time")
    options = EvaluationOptions(
        nonlinear=RootOptions(
            variables={"x": VariableOptions(initial=1, scale=1)},
            residual_scales={relation: 1},
        )
    )
    with (
        Implementation(Scalar, [relation]).with_options(options),
        pytest.raises(BiltAnalysisError, match="scalar"),
    ):
        _ = Scalar(p=tensor(np.array([6.0, 12.0]), over=(time,))).x


def test_initial_values_do_not_complete_an_underconstrained_model() -> None:
    options = _options()
    assert options.nonlinear is not None
    options = dataclasses.replace(
        options,
        nonlinear=dataclasses.replace(
            options.nonlinear,
            variables={
                **options.nonlinear.variables,
                "area": VariableOptions(initial=Quantity(6, meter**2), scale=Quantity(1, meter**2)),
            },
        ),
    )
    with _model().with_options(options), pytest.raises(BiltAnalysisError):
        _ = Rectangle(perimeter=Quantity(10, meter)).width


def test_configured_view_retains_options_for_new_attribute_reads_after_context_exit() -> None:
    model = _model()
    options = _options()
    configured = model.with_options(options)
    with configured:
        rectangle = _rectangle()
        width = rectangle.width
    assert rectangle.width is width
    # This output was never requested inside the context. It still uses the
    # configured implementation retained by the first successful calculation.
    assert rectangle.height.value == pytest.approx(3)
    assert rectangle.diagonal_squared.value == pytest.approx(13)
    assert model.options is None
    assert configured.options is options


def test_views_share_structure_without_mutating_or_registering_the_base() -> None:
    model = _model()
    registry_size = len(Rectangle.__bilt_system_registry__)
    first_options = _options()
    second_options = _options(width=4, height=1)
    first = model.with_options(first_options)
    second = first.with_options(second_options)
    assert model.options is None
    assert first.options is first_options
    assert second.options is second_options
    assert first.system is model.system
    assert second.structural_analysis() is model.structural_analysis()
    assert first.structural_analysis() is model.structural_analysis()
    assert len(Rectangle.__bilt_system_registry__) == registry_size
    assert first != second
    assert first != model
    # Structurally identical views using the same immutable policy are equivalent.
    assert model.with_options(first_options) == first


def test_supplied_values_do_not_pin_an_implementation_or_configuration() -> None:
    rectangle = Rectangle(width=Quantity(2, meter), height=Quantity(3, meter))
    model = _model()
    with model.with_options(EvaluationOptions()):
        assert rectangle.width.value == pytest.approx(2)
    # Reading an already supplied value needed no implementation binding.
    with pytest.raises(AttributeError):
        _ = rectangle.area
    with model:
        assert rectangle.area.value == pytest.approx(6)


def test_failed_requests_do_not_pin_a_failed_configuration() -> None:
    model = _model()
    options = _options()
    assert options.nonlinear is not None
    failed = model.with_options(
        dataclasses.replace(
            options, nonlinear=dataclasses.replace(options.nonlinear, controls=RootControls(max_evaluations=1))
        )
    )
    rectangle = _rectangle()
    with failed, pytest.raises(BiltNonlinearSolveError):
        _ = rectangle.width
    with model.with_options(options):
        assert rectangle.width.value == pytest.approx(2)
    assert rectangle.height.value == pytest.approx(3)


def test_cached_answers_are_owned_by_the_implementation_not_just_its_options() -> None:
    @system
    class Sample(metaclass=Interface):
        x: float = field()
        answer: float = field()

    def scaled(factor: float) -> typing.Any:  # noqa: ANN401
        @equation
        def relation(x: float, answer: float) -> bool:
            return answer == factor * x

        return relation

    options = EvaluationOptions()
    first = Implementation(Sample, [scaled(2)]).with_options(options)
    second = Implementation(Sample, [scaled(3)]).with_options(options)
    sample = Sample(x=5)
    with first:
        original = sample.answer
        assert original == pytest.approx(10)
        with second:
            assert sample.answer == pytest.approx(15)
        assert sample.answer == original
    assert sample.answer == original


def test_unconfigured_model_does_not_inherit_a_configured_implementations_settings() -> None:
    model = _model()
    configured = model.with_options(_options())
    rectangle = _rectangle()
    with configured:
        assert rectangle.width.value == pytest.approx(2)
        with model, pytest.raises(BiltAnalysisError):
            _ = rectangle.height
        assert rectangle.height.value == pytest.approx(3)
    assert rectangle.width.value == pytest.approx(2)


class _AlternatingRoots:
    """Two valid rectangle roots, exposing accidental repeated numerical solves."""

    name = "alternating-roots"
    backend = "python"

    def __init__(self) -> None:
        self.calls = 0

    def solve(self, residual: object, initial_guess: object, *, controls: RootControls) -> NonlinearResult:
        _ = residual, initial_guess, controls
        self.calls += 1
        # Unknowns are sorted height, width; both scales are one metre.
        candidate = np.array([3.0, 2.0] if self.calls % 2 else [2.0, 3.0])
        return NonlinearResult(solution=candidate, converged=True, message="valid rectangle root")


def _alternating_options(solver: _AlternatingRoots) -> EvaluationOptions:
    options = _options()
    assert options.nonlinear is not None
    return dataclasses.replace(options, nonlinear=dataclasses.replace(options.nonlinear, solver=solver))


@pytest.mark.parametrize("order", (("width", "height"), ("height", "width")))
def test_separate_output_reads_reuse_one_complete_nonlinear_solution(order: tuple[str, str]) -> None:
    solver = _AlternatingRoots()
    with _model().with_options(_alternating_options(solver)):
        instance = _rectangle()
        results = {name: getattr(instance, name).convert_to(meter).value for name in order}
        assert results["width"] == pytest.approx(2)
        assert results["height"] == pytest.approx(3)
        assert instance.diagonal_squared.convert_to(meter**2).value == pytest.approx(13)
    assert solver.calls == 1


def test_distinct_base_owners_are_isolated_but_views_of_one_owner_share_answers() -> None:
    solver = _AlternatingRoots()
    options = _alternating_options(solver)
    first_base = _model()
    first = first_base.with_options(options)
    second = _model().with_options(options)
    instance = _rectangle()
    with first:
        assert instance.width.convert_to(meter).value == pytest.approx(2)
    with second:
        assert instance.width.convert_to(meter).value == pytest.approx(3)
    with first_base.with_options(options):
        assert instance.height.convert_to(meter).value == pytest.approx(3)
    assert solver.calls == 2  # noqa: PLR2004 - two distinct owners or instances


def _wrap_child(child: Implementation) -> Implementation:
    @equation
    def supply_area(area: Area, inner: Rectangle) -> bool:
        return area == inner.area

    @equation
    def supply_perimeter(perimeter: Length, inner: Rectangle) -> bool:
        return perimeter == inner.perimeter

    @equation
    def expose_width(width: Length, inner: Rectangle) -> bool:
        return width == inner.width

    @equation
    def expose_height(height: Length, inner: Rectangle) -> bool:
        return height == inner.height

    return Implementation(
        Rectangle,
        [supply_area, supply_perimeter, expose_width, expose_height],
        internal_implementations={"inner": child},
    )


@pytest.mark.parametrize("depth", (1, 2))
def test_configured_child_solves_locally_under_unconfigured_parents(depth: int) -> None:
    solver = _AlternatingRoots()
    model = _model().with_options(_alternating_options(solver))
    for _ in range(depth):
        model = _wrap_child(model)
    with model:
        instance = _rectangle()
        assert instance.width.convert_to(meter).value == pytest.approx(2)
        assert instance.height.convert_to(meter).value == pytest.approx(3)
    assert solver.calls == 1


def test_parent_policy_does_not_replace_child_policy() -> None:
    child_solver = _AlternatingRoots()
    child = _model().with_options(_alternating_options(child_solver))
    parent = _wrap_child(child).with_options(EvaluationOptions(nonlinear=RootOptions(variables={}, residual_scales={})))
    with parent:
        assert _rectangle().width.convert_to(meter).value == pytest.approx(2)
    assert child_solver.calls == 1


@pytest.mark.parametrize("invalid", ("port", "equation"))
def test_child_preparation_validates_its_own_options(invalid: str) -> None:
    options = _options()
    assert options.nonlinear is not None
    if invalid == "port":
        roots = dataclasses.replace(
            options.nonlinear, variables={**options.nonlinear.variables, "typo": VariableOptions(initial=1, scale=1)}
        )
        match = "typo"
    else:

        @equation
        def unrelated(width: Length, height: Length) -> bool:
            return width == height

        roots = dataclasses.replace(
            options.nonlinear, residual_scales={**options.nonlinear.residual_scales, unrelated: Quantity(1, meter)}
        )
        match = "Residual scales"
    child = _model().with_options(dataclasses.replace(options, nonlinear=roots))
    with _wrap_child(child), pytest.raises(BiltAnalysisError, match=match):
        _ = _rectangle().width


def test_repr_and_equality_use_selected_known_values_without_solving() -> None:
    solver = _AlternatingRoots()
    model = _model().with_options(_alternating_options(solver))
    first = _rectangle()
    second = _rectangle()
    with model:
        assert first == second
        _ = first.width
        assert "width=" in repr(first)
        assert first != second
        _ = second.width
        assert first != second  # Same inputs, different known roots.
    assert solver.calls == 2  # noqa: PLR2004 - two distinct owners or instances


def test_repr_returns_to_outer_owner_on_a_cache_hit() -> None:
    solver = _AlternatingRoots()
    base = _model()
    first = base.with_options(_alternating_options(solver))
    second = base.with_options(_alternating_options(solver))
    instance = _rectangle()
    with first:
        width = instance.width
        outer_repr = repr(instance)
        with second:
            assert instance.width != width
            assert repr(instance) != outer_repr
        assert instance.width == width
        assert repr(instance) == outer_repr
    assert solver.calls == 2  # noqa: PLR2004 - two distinct owners or instances


def test_documented_nonlinear_example_executes(tmp_path: Path) -> None:
    document = Path(__file__).parents[2] / "docs/guide/nonlinear-solves.md"
    snippets = re.findall(r"```python\n(.*?)```", document.read_text(), re.DOTALL)
    script = tmp_path / "nonlinear_example.py"
    script.write_text("\n\n".join(snippets))
    namespace = runpy.run_path(str(script))
    assert namespace["width"].convert_to(meter).value == pytest.approx(2)
    assert namespace["height"].convert_to(meter).value == pytest.approx(3)
    with namespace["alternative_model"]:
        instance = namespace["Rectangle"](area=Quantity(6, meter**2), perimeter=Quantity(10, meter))
        assert instance.width.convert_to(meter).value == pytest.approx(2)


def test_sibling_children_keep_complete_solutions_separate() -> None:
    solver = _AlternatingRoots()
    child = _model().with_options(_alternating_options(solver))

    @equation
    def left_area(area: Area, left: Rectangle) -> bool:
        return left.area == area

    @equation
    def right_area(area: Area, right: Rectangle) -> bool:
        return right.area == area

    @equation
    def left_perimeter(perimeter: Length, left: Rectangle) -> bool:
        return left.perimeter == perimeter

    @equation
    def right_perimeter(perimeter: Length, right: Rectangle) -> bool:
        return right.perimeter == perimeter

    @equation
    def left_width(width: Length, left: Rectangle) -> bool:
        return width == left.width

    @equation
    def right_width(height: Length, right: Rectangle) -> bool:
        return height == right.width

    @equation
    def remaining_heights(diagonal_squared: Area, left: Rectangle, right: Rectangle) -> bool:
        return diagonal_squared == left.height * right.height

    parent = Implementation(
        Rectangle,
        [left_area, right_area, left_perimeter, right_perimeter, left_width, right_width, remaining_heights],
        internal_implementations={"left": child, "right": child},
    )
    with parent:
        instance = _rectangle()
        assert instance.width.convert_to(meter).value == pytest.approx(2)
        assert instance.height.convert_to(meter).value == pytest.approx(3)
        assert instance.diagonal_squared.convert_to(meter**2).value == pytest.approx(6)
    assert solver.calls == 2  # noqa: PLR2004 - one solve per child instance


@pytest.mark.parametrize("configured", (False, True))
def test_known_values_follow_ordinary_and_empty_configured_models(*, configured: bool) -> None:
    @system
    class Value(metaclass=Interface):
        source: float = field()
        answer: float = field()

    @equation
    def first(source: float, answer: float) -> bool:
        return answer == source + 1

    @equation
    def second(source: float, answer: float) -> bool:
        return answer == source + 2

    outer = Implementation(Value, [first])
    inner = Implementation(Value, [second])
    if configured:
        outer = outer.with_options(EvaluationOptions())
        inner = inner.with_options(EvaluationOptions())
    instance = Value(source=1.0)
    with outer:
        assert instance.answer == pytest.approx(2)
        expected = repr(instance)
        with inner:
            assert instance.answer == pytest.approx(3)
            assert repr(instance) != expected
        assert instance.answer == pytest.approx(2)
        assert repr(instance) == expected
