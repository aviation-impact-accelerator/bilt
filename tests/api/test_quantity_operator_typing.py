"""Check concrete equation operation typing and the generic system contract.

Use existing Quantity arithmetic and QuantityTensor reductions directly,
without auxiliary capability protocols or type-checker suppressions.
"""

import typing

import numpy as np

from bilt import (
    Implementation,
    Interface,
    Quantity,
    QuantityLike,
    QuantityTensor,
    Tensor,
    dim,
    equation,
    field,
    system,
    tensor,
)
from bilt.core._port_declaration import PortDeclaration
from bilt.symbolic._expressions import Expr, TensorExpr
from bilt.units import kilogram
from bilt.units.dimensions import MASS

type ScalarMass = typing.Annotated[Quantity, MASS]
type TensorMass = typing.Annotated[QuantityTensor, MASS]
type Mass = typing.Annotated[QuantityLike, MASS]

time = dim("time")
scenario = dim("scenario")


@system
class Scale(metaclass=Interface):
    mass: Mass = field()
    doubled: Mass = field()


@equation
def double_mass(mass: ScalarMass, doubled: ScalarMass) -> bool:
    """Concrete scalar arithmetic stays liftable in an equation."""
    return doubled == 2.0 * mass


@system
class Totals(metaclass=Interface):
    masses: TensorMass = field()
    total: Mass = field()


@equation
def total_mass(masses: TensorMass, total: ScalarMass) -> bool:
    """Tensor methods are available without a cast, ignore or getattr."""
    return total == masses.sum(time)


def _scalar_operations(mass: Quantity) -> Quantity:
    """Reuse the concrete scalar arithmetic API rather than a duplicate protocol."""
    result = -(mass + mass) * 2.0 / 2.0
    typing.assert_type(result, Quantity)
    return result


def _tensor_operations(mass: QuantityTensor) -> Tensor:
    """Named reductions retain the existing Tensor return type."""
    result = mass.sum(time)
    typing.assert_type(result, Tensor)
    return result


def test_native_scalar_satisfies_scalar_operations() -> None:
    original = Quantity(2.0, kilogram)
    result = _scalar_operations(original)
    assert result == -4.0 * kilogram


def test_quantity_tensor_satisfies_tensor_operations() -> None:
    masses = tensor([1.0, 2.0], over=time, unit=kilogram)
    typing.assert_type(masses, QuantityTensor)
    assert isinstance(masses, Tensor)
    # Type-check the reduction without executing it on a concrete Tensor.
    if typing.TYPE_CHECKING:
        _tensor_operations(masses)


def test_scalar_arithmetic_equation_lifts_without_type_ignores() -> None:
    assert type(double_mass.__bilt_equation_symbols__["mass"]) is Expr
    assert double_mass.__bilt_port_declarations__["mass"] == PortDeclaration(MASS)
    assert Scale.__bilt_port_declarations__["mass"] == PortDeclaration(MASS)

    masses = tensor([1.0, 2.0, 3.0], over=time, unit=kilogram)
    with Implementation(Scale, [double_mass]):
        result = Scale(mass=masses).doubled

    assert isinstance(result, Tensor)
    np.testing.assert_array_equal(result.values, [2.0, 4.0, 6.0])


def test_scalar_arithmetic_equation_with_scalar_input() -> None:
    with Implementation(Scale, [double_mass]):
        result = Scale(mass=Quantity(2.0, kilogram)).doubled
    assert result == 4.0 * kilogram


def test_tensor_reduction_equation_retains_outer_axis() -> None:
    assert isinstance(total_mass.__bilt_equation_symbols__["masses"], TensorExpr)
    assert total_mass.__bilt_port_declarations__["masses"].over_a_domain
    masses = tensor([[1.0, 2.0], [3.0, 4.0]], over=(scenario, time), unit=kilogram)
    with Implementation(Totals, [total_mass]):
        result = Totals(masses=masses).total
    assert isinstance(result, Tensor)
    assert result.domain.dims == (scenario,)
    np.testing.assert_array_equal(result.values, [3.0, 7.0])
