"""Quantity annotations and a working Bilt system/implementation example.

The ABC annotations below test the proposed Python contract. The decorated
system uses *existing* Bilt annotations until the compiler supports the ABC.
"""

from typing import Annotated

from bilt import Implementation, Interface, Tensor, equation, field, system
from bilt.units import Quantity as NativeQuantity
from bilt.units import kilogram
from bilt.units.dimensions import MASS

from .prototype import Quantity, QuantityTensor, ScalarQuantity

type Mass = Annotated[Quantity, MASS]
type TensorMass = Annotated[Quantity[Tensor], MASS]
type NativeMass = Annotated[NativeQuantity, MASS]


def system_boundary(mass: Mass) -> object:
    """Bare Quantity accepts either concrete magnitude."""
    return mass.magnitude


def tensor_equation(mass: TensorMass) -> tuple[int, ...]:
    """Tensor-aware equations can request a tensor magnitude."""
    return mass.magnitude.shape


def scalar_equation(mass: ScalarQuantity) -> float:
    return mass.magnitude


def both_interfaces(mass: QuantityTensor) -> tuple[int, ...]:
    return mass.shape


@system
class MassScale(metaclass=Interface):
    """Executable model using Bilt's current annotation representation."""

    mass: NativeMass = field()
    doubled: NativeMass = field()


@equation
def double_mass(mass: NativeMass, doubled: NativeMass) -> bool:
    return mass + mass == doubled


mass_scale_implementation = Implementation(MassScale, [double_mass])


def run_mass_scale() -> NativeQuantity:
    """A real model run through Bilt's structural analysis and evaluator."""
    with mass_scale_implementation:
        model = MassScale(mass=2.0 * kilogram)
        return model.doubled
