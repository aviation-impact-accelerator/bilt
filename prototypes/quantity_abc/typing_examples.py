"""Examples to evaluate with ty and Pyrefly.

These are intentionally functions, not executable model code, because the
prototype is not yet hooked into Bilt's equation annotation parser.
"""

from typing import Annotated

from bilt import Tensor
from bilt.units.dimensions import MASS
from prototypes.quantity_abc.prototype import Quantity, QuantityTensor, ScalarQuantity

type Mass = Annotated[Quantity, MASS]
type TensorMass = Annotated[Quantity[Tensor], MASS]


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
