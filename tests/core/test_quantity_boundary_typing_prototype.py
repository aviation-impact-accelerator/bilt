"""Prototype: typed quantity system inputs using Bilt's real constructor/evaluator.

Only the annotation adapter and concrete tensor-quantity factory are experimental.
Production Bilt code is unchanged, and positive examples have no type suppressions.
"""

import inspect
import typing
from typing import Annotated, Protocol, assert_type

import numpy as np

from bilt import Implementation, Interface, Tensor, dim, equation, field, system, tensor
from bilt.core._port_declaration import annotated_parts
from bilt.symbolic._expressions import Expr
from bilt.units import Quantity as NativeQuantity
from bilt.units import Unit, gram, kilogram
from bilt.units.dimensions import MASS

if typing.TYPE_CHECKING:
    from bilt.symbolic._dimensions import Dim
    from bilt.symbolic._domains import Domain


class QuantityLike[M: float | Tensor = float | Tensor](Protocol):
    """Small *structural* quantity contract satisfied by existing scalar values.

    The public value/unit properties are present on native Bilt Quantity;
    unlike the nominal alternatives in #390 this requires no subclassing of
    Bilt's existing final scalar Quantity.
    """

    @property
    def value(self) -> M:
        """Return an unlabelled magnitude."""

    @property
    def unit(self) -> Unit:
        """Return a non-null unit."""


class QuantityTensor(Tensor):
    """Unit-bearing Tensor exposing the shared quantity value/unit interface."""

    unit: Unit

    def __init__(self, values: np.ndarray, domain: Domain, unit: Unit) -> None:
        super().__init__(values=values, domain=domain, unit=unit)

    @property
    def value(self) -> Tensor:
        """Return the tensor magnitude without its unit."""
        return Tensor(values=self.values, domain=self.domain)


def quantity_tensor(values: object, *, over: Dim, unit: Unit) -> QuantityTensor:
    """Genuinely typed unit-bearing tensor factory, with no cast or Any."""
    initial = tensor(values, over=over, unit=unit)
    return QuantityTensor(values=initial.values, domain=initial.domain, unit=unit)


type Mass = Annotated[QuantityLike, MASS]
type ScalarMass = Annotated[QuantityLike[float], MASS]
type TensorMass = Annotated[QuantityLike[Tensor], MASS]


def _native_annotation(annotation: object) -> object:
    """Translate prototype contracts into production Bilt port declarations."""
    base, metadata = annotated_parts(annotation)
    if base is QuantityLike:
        return Annotated[NativeQuantity, *metadata] if metadata else NativeQuantity
    if typing.get_origin(base) is not QuantityLike:
        return annotation
    if typing.get_args(base) == (Tensor,):
        return Annotated[Tensor, *metadata] if metadata else Tensor
    if typing.get_args(base) == (float,):
        return Annotated[NativeQuantity, *metadata] if metadata else NativeQuantity
    msg = f"Unsupported quantity annotation: {base!r}"
    raise TypeError(msg)


def prototype_system[T: Interface](cls: T) -> T:
    """Reuse Bilt's real constructor and evaluator after annotation translation."""
    original = cls.__annotations__
    resolved = inspect.get_annotations(cls, eval_str=True)
    cls.__annotations__ = {name: _native_annotation(annotation) for name, annotation in resolved.items()}
    try:
        return system(cls)
    finally:
        cls.__annotations__ = original


@prototype_system
class MassScale(metaclass=Interface):
    """A generic quantity system field accepts either concrete magnitude."""

    mass: Mass = field()
    doubled: Mass = field()


@equation
def doubling(mass: Annotated[NativeQuantity, MASS], doubled: Annotated[NativeQuantity, MASS]) -> bool:
    """Native scalar equation which Bilt automatically lifts across domains."""
    return doubled == mass * 2.0


def accepts_mass(mass: Mass) -> Unit:
    """Positive checker fixture: structural contract works without casting."""
    return mass.unit


def accepts_scalar(mass: ScalarMass) -> float:
    """Positive checker fixture: scalar quantity magnitude is float."""
    return mass.value


def accepts_tensor(mass: TensorMass) -> tuple[int, ...]:
    """Positive checker fixture: tensor magnitude exposes its dimensions."""
    return mass.value.shape


def test_native_quantity_and_tensor_conform_to_generic_contract() -> None:
    """These calls must be valid to both Pyrefly and ty with no ignores."""
    scalar = NativeQuantity(2.0, kilogram)
    varied = quantity_tensor([1.0, 2.0, 3.0], over=dim("time"), unit=kilogram)

    assert_type(scalar.value, float)
    assert_type(varied.value, Tensor)
    assert accepts_mass(scalar) is kilogram
    assert accepts_mass(varied) is kilogram
    assert accepts_scalar(scalar) == scalar.value
    assert accepts_tensor(varied) == (3,)
    assert varied.value.unit is None
    assert isinstance(varied, Tensor)


def test_bilt_constructor_accepts_both_magnitudes_without_suppressions() -> None:
    """Real Interface dataclass_transform constructor, not a stand-in class."""
    time = dim("time")
    scalar = NativeQuantity(2.0, kilogram)
    varied = quantity_tensor([1.0, 2.0, 3.0], over=time, unit=kilogram)

    with Implementation(MassScale, [doubling]):
        scalar_result = MassScale(mass=scalar).doubled
        tensor_result = MassScale(mass=varied).doubled

    assert scalar_result == 4.0 * kilogram
    assert isinstance(tensor_result, Tensor)
    assert tensor_result.unit is kilogram
    assert tensor_result.domain.dims == (time,)
    np.testing.assert_array_equal(tensor_result.values, [2.0, 4.0, 6.0])


def test_mixed_scalar_tensor_inputs_still_lift() -> None:
    """A second real equation can lift a scalar offset over a tensor input."""

    @prototype_system
    class AddedMass(metaclass=Interface):
        mass: Mass = field()
        extra: Mass = field()
        total: Mass = field()

    @equation
    def total(\n        mass: Annotated[NativeQuantity, MASS],\n        extra: Annotated[NativeQuantity, MASS],\n        total: Annotated[NativeQuantity, MASS],\n    ) -> bool:
        return total == mass + extra

    with Implementation(AddedMass, [total]):
        answer = AddedMass(
            mass=quantity_tensor([1.0, 2.0], over=dim("time"), unit=kilogram),
            extra=NativeQuantity(500.0, gram),
        ).total

    assert isinstance(answer, Tensor)
    np.testing.assert_array_equal(answer.values, [1.5, 2.5])


def test_real_equation_remains_scalar_local() -> None:
    """The compiler still sees a plain Expr, eligible for lifting."""
    assert type(doubling.__bilt_equation_symbols__["mass"]) is Expr
