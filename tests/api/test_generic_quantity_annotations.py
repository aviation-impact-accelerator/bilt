"""Generic quantity annotations use Bilt's real compiler and lifting.

This tests the annotation and value-boundary semantics. Typing operator
signatures for the shared protocol will be refined in a separate PR.
"""

import typing

import numpy as np
import pytest

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
from bilt.errors import BiltInputDimensionalityError, BiltPortDeclarationError
from bilt.symbolic._expressions import Expr, TensorExpr
from bilt.units import kilogram, meter
from bilt.units.dimensions import MASS

type Mass = typing.Annotated[QuantityLike, MASS]
type ScalarMass = typing.Annotated[QuantityLike[float], MASS]
type TensorMass = typing.Annotated[QuantityLike[Tensor], MASS]
time = dim("time")
scenario = dim("scenario")


@system
class Scale(metaclass=Interface):
    mass: Mass = field()
    doubled: Mass = field()


@equation
def copy_mass(mass: ScalarMass, doubled: ScalarMass) -> bool:
    """The scalar-local equation can be lifted over any input domain."""
    return doubled == mass


@system
class ScalarOnly(metaclass=Interface):
    mass: ScalarMass = field()


@system
class TensorOnly(metaclass=Interface):
    mass: TensorMass = field()
    total: Mass = field()


@equation
def reduce_mass(mass: TensorMass, total: ScalarMass) -> bool:
    """This reduction uses Bilt's tensor symbolic expression and real lowering.

    The generic protocol does not yet declare tensor-specific operators;
    adding their static signatures is a separate API decision.
    """
    return total == getattr(mass, "sum")(time)


def test_generic_quantity_declarations_distinguish_boundaries() -> None:
    assert Scale.__bilt_port_declarations__["mass"] == PortDeclaration(MASS)
    assert not Scale.__bilt_port_declarations__["mass"].scalar_only
    assert not Scale.__bilt_port_declarations__["mass"].over_a_domain

    assert ScalarOnly.__bilt_port_declarations__["mass"].scalar_only
    assert not ScalarOnly.__bilt_port_declarations__["mass"].over_a_domain

    assert TensorOnly.__bilt_port_declarations__["mass"].over_a_domain
    assert not TensorOnly.__bilt_port_declarations__["mass"].scalar_only


def test_scalar_local_equation_uses_scalar_symbols() -> None:
    assert type(copy_mass.__bilt_equation_symbols__["mass"]) is Expr
    assert not copy_mass.__bilt_port_declarations__["mass"].over_a_domain


def test_tensor_specific_equation_uses_tensor_symbols() -> None:
    assert isinstance(reduce_mass.__bilt_equation_symbols__["mass"], TensorExpr)
    assert reduce_mass.__bilt_port_declarations__["mass"].over_a_domain


def test_generic_system_field_accepts_scalar_quantity() -> None:
    with Implementation(Scale, [copy_mass]):
        result = Scale(mass=Quantity(2.0, kilogram)).doubled
    assert result == 2.0 * kilogram


def test_scalar_local_equation_lifts_tensor_quantity_without_type_ignores() -> None:
    masses = tensor([2.0, 3.0], over=time, unit=kilogram)
    typing.assert_type(masses, QuantityTensor)
    with Implementation(Scale, [copy_mass]):
        result = Scale(mass=masses).doubled

    assert isinstance(result, Tensor)
    assert result.domain.dims == (time,)
    np.testing.assert_array_equal(result.values, [2.0, 3.0])


def test_scalar_local_equation_lifts_across_two_dimensions() -> None:
    masses = tensor([[1.0, 2.0], [3.0, 4.0]], over=(time, scenario), unit=kilogram)
    with Implementation(Scale, [copy_mass]):
        result = Scale(mass=masses).doubled
    assert isinstance(result, Tensor)
    assert result.domain.dims == (time, scenario)
    np.testing.assert_array_equal(result.values, [[1.0, 2.0], [3.0, 4.0]])


def test_tensor_specific_reduction_uses_real_compiler() -> None:
    masses = tensor([2.0, 3.0], over=time, unit=kilogram)
    with Implementation(TensorOnly, [reduce_mass]):
        result = TensorOnly(mass=masses).total
    assert result == 5.0 * kilogram


def test_tensor_reduction_lifts_over_remaining_axis() -> None:
    masses = tensor([[2.0, 3.0], [4.0, 5.0]], over=(scenario, time), unit=kilogram)
    with Implementation(TensorOnly, [reduce_mass]):
        result = TensorOnly(mass=masses).total
    assert isinstance(result, Tensor)
    assert result.domain.dims == (scenario,)
    np.testing.assert_array_equal(result.values, [5.0, 9.0])


def test_scalar_only_boundary_rejects_tensor_at_runtime() -> None:
    masses = tensor([1.0, 2.0], over=time, unit=kilogram)
    # Bypass static typing intentionally to check the runtime guard.
    invalid = typing.cast("QuantityLike[float]", masses)
    with pytest.raises(BiltInputDimensionalityError):
        ScalarOnly(mass=invalid)


def test_tensor_only_boundary_rejects_scalar_at_runtime() -> None:
    # Likewise, an invalid value must be rejected when supplied dynamically.
    invalid = typing.cast("QuantityLike[Tensor]", Quantity(2.0, kilogram))
    with pytest.raises(BiltInputDimensionalityError):
        TensorOnly(mass=invalid)


def test_legacy_scalar_quantity_remains_liftable() -> None:
    """The existing annotations keep their permissive lifting semantics."""
    type LegacyMass = typing.Annotated[Quantity, MASS]

    @system
    class Legacy(metaclass=Interface):
        m: LegacyMass = field()

    assert not Legacy.__bilt_port_declarations__["m"].scalar_only


def test_dimension_mismatch_is_still_rejected() -> None:
    with pytest.raises(BiltInputDimensionalityError):
        Scale(mass=tensor([1.0], over=time, unit=meter))


def test_generic_annotation_still_requires_dimensionality() -> None:
    with pytest.raises(BiltPortDeclarationError, match="without a dimensionality"):

        @system
        class MissingDimension(metaclass=Interface):
            mass: QuantityLike = field()
