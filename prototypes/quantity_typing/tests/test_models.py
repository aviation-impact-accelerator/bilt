"""Exercise the actual Bilt compiler and evaluator through the prototype adapter."""

import numpy as np
import pytest
from bilt_typing import Implementation, Tensor, tensor, equal, system, field, Interface
from bilt_typing.quantities import Mass
from bilt_typing.units import (
    kilogram,
    gram,
    meter,
    meter_per_second_squared,
    absolute_celsius,
    absolute_fahrenheit,
    kilogram_per_meter,
)
from bilt_typing import dimension_types as dt
from bilt.units.dimensions import MASS
from examples import (
    AIRCRAFT,
    YEAR,
    Scale,
    ScalarScale,
    Fleet,
    FleetTotal,
    Cable,
    AbsoluteThermometer,
    doubling,
    newtons_law,
    total_mass,
    cable_mass,
    same_temperature,
    doubled_pointwise,
    _doubled,
)


def test_generated_markers_reuse_runtime_dimensions():
    assert dt.MASS.runtime is MASS


def test_scalar_system_conversion_and_reverse_equation():
    with Implementation(ScalarScale, [doubling]):
        assert ScalarScale(m=3000.0 * gram).doubled.convert_to(kilogram).value == 6.0
        assert ScalarScale(doubled=6.0 * kilogram).m.value == 3.0


def test_tensor_system_uses_real_lifting():
    result = _doubled(tensor([1.0, 2.0, 3.0], over=AIRCRAFT, unit=kilogram))
    np.testing.assert_allclose(result.value.values, [2, 4, 6])
    assert result.unit.dimension is dt.MASS


def test_mixed_scalar_tensor_equation():
    with Implementation(Fleet, [newtons_law]):
        result = Fleet(
            mass=tensor([1.0, 2.0], over=AIRCRAFT, unit=kilogram), acceleration=3.0 * meter_per_second_squared
        ).force
    np.testing.assert_allclose(result.value.values, [3, 6])
    assert result.unit.dimension is dt.FORCE


def test_absolute_temperature_equality():
    with Implementation(AbsoluteThermometer, [same_temperature]):
        result = AbsoluteThermometer(t=32.0 * absolute_fahrenheit).same
    assert result.convert_to(absolute_celsius).value == pytest.approx(0.0)


def test_reduction_final_axis():
    with Implementation(FleetTotal, [total_mass]):
        result = FleetTotal(masses=tensor([1.0, 2.0, 3.0], over=AIRCRAFT, unit=kilogram)).total
    assert result.value == 6.0


def test_reduction_retains_other_axes():
    with Implementation(FleetTotal, [total_mass]):
        result = FleetTotal(
            masses=tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], over=(YEAR, AIRCRAFT), unit=kilogram)
        ).total
    assert isinstance(result.value, Tensor)
    assert result.value.domain.dims == (YEAR,)
    np.testing.assert_allclose(result.value.values, [6, 15])


def test_missing_required_axis_rejected():
    with pytest.raises(TypeError, match="required tensor axis"):
        FleetTotal(masses=tensor([1.0, 2.0], over=YEAR, unit=kilogram))


def test_scalar_tensor_and_dimension_contracts_rejected():
    with pytest.raises(TypeError, match="representation"):
        Scale(m=3.0 * kilogram)
    with pytest.raises(TypeError, match="representation"):
        ScalarScale(m=tensor([1.0], over=AIRCRAFT, unit=kilogram))
    with pytest.raises(TypeError, match="Expected MASS"):
        ScalarScale(m=3.0 * meter)


def test_scalar_output_cannot_silently_be_tensor():
    @system
    class WrongScale(metaclass=Interface):
        m: Mass[Tensor[float]] = field()
        doubled: Mass[float] = field()

    with Implementation(WrongScale, [doubling]), pytest.raises(TypeError, match="representation"):
        WrongScale(m=tensor([1.0, 2.0], over=AIRCRAFT, unit=kilogram)).doubled


def test_pointwise_preserves_units_and_empty_domain():
    assert doubled_pointwise(3.0 * gram).value == 6.0
    result = doubled_pointwise(tensor([], over=AIRCRAFT, unit=gram))
    assert result.value.values.shape == (0,)
    assert result.unit is gram


def test_custom_dimension_arithmetic_and_model():
    density = 2.0 * kilogram_per_meter
    assert (density * (3.0 * meter)).value == 6.0
    with Implementation(Cable, [cable_mass]):
        assert Cable(density=density, length=3.0 * meter).mass.value == 6.0


def test_direct_arithmetic_mixed_and_tensor_operands():
    m = tensor([1.0, 2.0], over=AIRCRAFT, unit=kilogram)
    a = tensor([3.0, 4.0], over=AIRCRAFT, unit=meter_per_second_squared)
    np.testing.assert_allclose((m * a).value.values, [3, 8])
    np.testing.assert_allclose(((2.0 * kilogram) * a).value.values, [6, 8])
    np.testing.assert_allclose((m * (2.0 * meter_per_second_squared)).value.values, [2, 4])
    assert (m * a).unit.dimension is dt.FORCE


def test_quantity_addition_converts_units():
    assert ((1.0 * kilogram) + (500.0 * gram)).value == 1.5


def test_typed_equality_rejects_wrong_dimension_at_runtime():
    with pytest.raises(TypeError, match="Incompatible equation"):
        equal(1.0 * kilogram, 1.0 * meter)


def test_direct_arithmetic_rejects_incompatible_shapes():
    a = tensor([1.0, 2.0], over=AIRCRAFT, unit=kilogram)
    b = tensor([1.0, 2.0, 3.0], over=AIRCRAFT, unit=meter_per_second_squared)
    with pytest.raises(ValueError, match="matching tensor"):
        a * b


def test_whole_tensor_transform():
    from examples import total

    with Implementation(FleetTotal, [total]):
        result = FleetTotal(masses=tensor([[1.0, 2.0], [3.0, 4.0]], over=(YEAR, AIRCRAFT), unit=kilogram)).total
    assert result.value.domain.dims == (YEAR,)
    np.testing.assert_allclose(result.value.values, [3, 7])
