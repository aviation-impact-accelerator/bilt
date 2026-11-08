"""Intended quantity annotations plus self-contained executable model examples.

The mini-system below is an architectural fixture, not Bilt's production
`@system` decorator. The live Bilt model remains in ../typing_examples.py.
"""

from dataclasses import dataclass
from typing import Annotated

from bilt.units import kilogram
from bilt.units.dimensions import MASS

from ._quantity import Quantity
from ._quantity_tensor import QuantityTensor, typed
from ._tensor import Tensor, tensor

type Mass = Annotated[Quantity, MASS]
type TensorMass = Annotated[Quantity[Tensor], MASS]


@dataclass
class MassScale:
    """Proposed field signatures independent of a particular magnitude."""

    mass: Mass
    doubled: Mass | None = None


def double_mass(mass: Mass) -> Mass:
    """Scalar-safe equation body, suitable for lifting to tensor quantities."""
    return mass * 2.0


def total_mass(mass: TensorMass, axis: str) -> Mass:
    """Tensor-only body: requires a named-axis reduction."""
    reduced = mass.magnitude.sum(axis)
    if isinstance(reduced, Tensor):
        return QuantityTensor(reduced.values, reduced.domain, mass.unit)
    return Quantity(float(reduced), mass.unit)


class Implementation:
    """Minimal test wiring, not a substitute for Bilt's real Implementation."""

    def solve(self, model: MassScale) -> MassScale:
        """Evaluate a scalar-safe equation and save the result."""
        model.doubled = double_mass(model.mass)
        return model


def scalar_model() -> MassScale:
    """Use a concrete Quantity with the existing scalar constructor."""
    return Implementation().solve(MassScale(Quantity(2.0, kilogram)))


def tensor_model() -> MassScale:
    """Use the same annotated model and equation with a tensor magnitude."""
    input_mass = tensor([2.0, 3.0], over=("aircraft",)) * typed(kilogram)
    return Implementation().solve(MassScale(input_mass))
