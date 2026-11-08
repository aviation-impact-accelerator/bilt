"""Experiment: scalar equation annotations lift over tensor-valued system ports.

The experimental decorators only translate the *annotations* before invoking
Bilt's real system/equation compiler. No production compiler is replaced.
"""

import inspect
import typing

import numpy as np
import pytest

from bilt import Implementation, Interface, Tensor, dim, equation, field, system, tensor
from bilt.core._port_declaration import annotated_parts
from bilt.symbolic._expressions import Expr, TensorExpr
from bilt.units import Quantity as NativeQuantity
from bilt.units import gram, kilogram, meter
from bilt.units.dimensions import MASS

if typing.TYPE_CHECKING:
    from bilt.core._equation import BasicEquation
    from bilt.symbolic._dimensions import Dim


class Quantity[M: float | Tensor = float | Tensor]:
    """Annotation-only generic quantity, not a second runtime value type.

    Bilt never instantiates this object. The signatures represent operations
    that a production generic Quantity would need to expose to type checkers.
    """

    def __mul__(self, factor: float) -> Quantity[M]:
        """Represent scalar multiplication in a Bilt equation."""
        raise NotImplementedError

    def __add__(self, other: Quantity) -> Quantity[M]:
        """Represent ordinary quantity addition."""
        raise NotImplementedError

    def sum(self: Quantity[Tensor], axis: Dim) -> Quantity[float]:
        """Represent a tensor-only reduction in a Bilt equation."""
        raise NotImplementedError


type Mass = typing.Annotated[Quantity, MASS]
type ScalarMass = typing.Annotated[Quantity[float], MASS]
type TensorMass = typing.Annotated[Quantity[Tensor], MASS]


def _kind(annotation: object) -> typing.Literal["scalar", "tensor", "either"] | None:
    """Classify a proposed quantity annotation, unwrapping Annotated/aliases."""
    base, _ = annotated_parts(annotation)
    if base is Quantity:
        return "either"
    if typing.get_origin(base) is not Quantity:
        return None

    args = typing.get_args(base)
    if args == (float,):
        return "scalar"
    if args == (Tensor,):
        return "tensor"
    msg = f"Unsupported quantity magnitude parameter: {args!r}"
    raise TypeError(msg)


def _native_annotation(annotation: object) -> object:
    """Map a proposed annotation to Bilt's existing symbolic declaration."""
    kind = _kind(annotation)
    if kind is None:
        return annotation

    _, metadata = annotated_parts(annotation)
    native = Tensor if kind == "tensor" else NativeQuantity
    return typing.Annotated[native, *metadata] if metadata else native


def prototype_equation[R, **P](function: typing.Callable[P, R]) -> BasicEquation[R, P]:
    """Build a real Bilt equation from proposed generic annotations.

    The original function annotations are restored after compilation. Only
    Bilt's existing symbolic expression selection and dimensional checks run.
    """
    original = function.__annotations__
    resolved = inspect.get_annotations(function, eval_str=True)
    function.__annotations__ = {name: _native_annotation(annotation) for name, annotation in resolved.items()}
    try:
        return equation(function)
    finally:
        function.__annotations__ = original


def prototype_system[T: Interface](cls: T) -> T:
    """Build a real Bilt system with translated internal port declarations.

    Preserve the original annotations for static checkers and the optional
    prototype boundary validator below.
    """
    original = cls.__annotations__
    resolved = inspect.get_annotations(cls, eval_str=True)
    cls.__annotations__ = {name: _native_annotation(annotation) for name, annotation in resolved.items()}
    try:
        return system(cls)
    finally:
        cls.__annotations__ = original


def prototype_instantiate(cls: Interface, **inputs: object) -> object:
    """Experiment with strict boundary types without changing Bilt constructors.

    Bilt still performs its own existing unit validation. This is an opt-in
    boundary check only, not behaviour installed by prototype_system.
    """
    for name, annotation in inspect.get_annotations(cls, eval_str=True).items():
        if name not in inputs:
            continue
        kind = _kind(annotation)
        value = inputs[name]
        if kind == "scalar" and isinstance(value, Tensor):
            msg = f"{name} requires a scalar quantity at the system boundary"
            raise TypeError(msg)
        if kind == "tensor" and not isinstance(value, Tensor):
            msg = f"{name} requires a tensor quantity at the system boundary"
            raise TypeError(msg)
    return typing.cast("Callable[..., object]", cls)(**inputs)


@prototype_system
class Scale(metaclass=Interface):
    """The same quantity field should accept scalar or tensor-valued inputs."""

    m: Mass = field()
    doubled: Mass = field()


@prototype_equation
def double_mass(m: ScalarMass, doubled: ScalarMass) -> bool:
    """Scalar-local equation, independently of actual system input domain."""
    return doubled == m * 2.0


@prototype_system
class AddMass(metaclass=Interface):
    """Allow scalar and tensor inputs to an ordinary scalar equation."""

    m: Mass = field()
    extra: Mass = field()
    total: Mass = field()


@prototype_equation
def add_mass(m: ScalarMass, extra: ScalarMass, total: ScalarMass) -> bool:
    """The equation does not explicitly deal with any tensor dimensions."""
    return total == m + extra


time = dim("time")
scenario = dim("scenario")


@prototype_system
class Totals(metaclass=Interface):
    """The inner tensor domain is required but extra outer axes are allowed."""

    m: TensorMass = field()
    total: Mass = field()


@prototype_equation
def total_mass(m: TensorMass, total: ScalarMass) -> bool:
    """Reduce the inner time axis and lift across remaining outer axes."""
    return total == m.sum(time)


@prototype_system
class StrictBoundary(metaclass=Interface):
    """A system boundary explicitly requiring a scalar quantity."""

    m: ScalarMass = field()


def test_proposed_annotation_kinds_are_distinct() -> None:
    """Classify bare, scalar and tensor annotations before translation."""
    assert _kind(Mass) == "either"
    assert _kind(ScalarMass) == "scalar"
    assert _kind(TensorMass) == "tensor"
    assert _kind(int) is None


def test_scalar_local_equations_get_plain_symbols() -> None:
    """A scalar-local annotation must not create a TensorExpr."""
    assert type(double_mass.__bilt_equation_symbols__["m"]) is Expr
    assert type(double_mass.__bilt_equation_symbols__["doubled"]) is Expr
    assert not double_mass.__bilt_port_declarations__["m"].over_a_domain


def test_tensor_specific_equation_gets_tensor_symbol_and_domain_declaration() -> None:
    """The explicit tensor annotation must select the actual TensorExpr."""
    assert isinstance(total_mass.__bilt_equation_symbols__["m"], TensorExpr)
    assert total_mass.__bilt_port_declarations__["m"].over_a_domain
    assert Totals.__bilt_port_declarations__["m"].over_a_domain


def test_scalar_quantity_input_uses_real_bilt_compiler() -> None:
    """A scalar result should remain a native Bilt Quantity."""
    with Implementation(Scale, [double_mass]):
        result = Scale(m=2.0 * kilogram).doubled
    assert result == 4.0 * kilogram


def test_scalar_local_quantity_equation_lifts_across_one_tensor_axis() -> None:
    """The same compiled scalar-local equation lifts over a unit-bearing tensor."""
    masses = tensor([1.0, 2.0, 3.0], over=time, unit=kilogram)
    with Implementation(Scale, [double_mass]):
        result = Scale(m=masses).doubled  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]
    assert isinstance(result, Tensor)
    assert result.domain.dims == (time,)
    assert result.unit is kilogram
    np.testing.assert_array_equal(result.values, [2.0, 4.0, 6.0])


def test_scalar_and_tensor_quantities_mix_in_one_scalar_local_equation() -> None:
    """Bilt applies the unit conversion and lifts a scalar quantity."""
    with Implementation(AddMass, [add_mass]):
        result = AddMass(
            m=tensor([1.0, 2.0], over=time, unit=kilogram),  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]
            extra=500.0 * gram,
        ).total
    assert isinstance(result, Tensor)
    assert result.unit is kilogram
    np.testing.assert_array_equal(result.values, [1.5, 2.5])


def test_scalar_equation_broadcasts_across_disjoint_tensor_domains() -> None:
    """Lifting uses the union of both concrete input domains."""
    with Implementation(AddMass, [add_mass]):
        result = AddMass(
            m=tensor([1.0, 2.0], over=time, unit=kilogram),  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]
            extra=tensor([10.0, 20.0, 30.0], over=scenario, unit=kilogram),  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]
        ).total
    assert isinstance(result, Tensor)
    assert result.domain.dims == (time, scenario)
    np.testing.assert_array_equal(result.values, [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]])


def test_tensor_specific_reduction_keeps_outer_lifted_axis() -> None:
    """Tensor-local reductions keep Bilt's outer broadcasting machinery."""
    masses = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], over=(scenario, time), unit=kilogram)
    with Implementation(Totals, [total_mass]):
        result = Totals(m=masses).total  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]
    assert isinstance(result, Tensor)
    assert result.unit is kilogram
    assert result.domain.dims == (scenario,)
    np.testing.assert_array_equal(result.values, [6.0, 15.0])


def test_scalar_annotation_is_strict_only_at_explicit_boundary() -> None:
    """Boundary restrictions differ from local equation lifting."""
    with pytest.raises(TypeError, match="requires a scalar quantity"):
        prototype_instantiate(StrictBoundary, m=tensor([1.0, 2.0], over=time, unit=kilogram))
    result = prototype_instantiate(StrictBoundary, m=2.0 * kilogram)
    assert isinstance(result, StrictBoundary)


def test_tensor_annotation_requires_tensor_at_explicit_boundary() -> None:
    """A declared tensor port should not accept a scalar quantity."""
    with pytest.raises(TypeError, match="requires a tensor quantity"):
        prototype_instantiate(Totals, m=2.0 * kilogram)


def test_generic_boundary_accepts_either_magnitude() -> None:
    """The generic boundary is not as restrictive as either specialisation."""
    scalar = prototype_instantiate(Scale, m=2.0 * kilogram)
    varied = prototype_instantiate(Scale, m=tensor([1.0, 2.0], over=time, unit=kilogram))
    assert isinstance(scalar, Scale)
    assert isinstance(varied, Scale)


def test_invalid_generic_parameter_is_rejected_by_experiment() -> None:
    """Explicitly reject generics beyond the scalar/tensor experiment."""
    with pytest.raises(TypeError, match="Unsupported quantity magnitude"):
        _kind(Quantity[str])  # type: ignore[type-var]  # pyrefly: ignore[bad-type-argument]  # ty: ignore[invalid-type-arguments]


def test_production_annotated_equation_remains_unchanged() -> None:
    """This adapter never modifies Bilt's production annotation parser."""
    assert _native_annotation(typing.Annotated[NativeQuantity, MASS]) == typing.Annotated[NativeQuantity, MASS]
    assert _native_annotation(typing.Annotated[Tensor, MASS]) == typing.Annotated[Tensor, MASS]
    assert _native_annotation(meter) is meter
