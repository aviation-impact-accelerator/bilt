"""A declared port holds a unit-carrying tensor, checked and lowered at the boundary and labelled on the way out."""

import typing

import numpy as np
import pytest

from bilt import (
    JAX_BACKEND,
    PYTHON_BACKEND,
    Implementation,
    Interface,
    Quantity,
    QuantityTensor,
    Tensor,
    dim,
    equation,
    field,
    system,
    tensor,
    transform,
)
from bilt.errors import BiltImplementationDimensionalityError, BiltInputDimensionalityError
from bilt.units import Metadata, absolute_celsius, gram, kelvin, kilogram, meter, second, tonne
from bilt.units.dimensions import ABSOLUTE, FORCE, LENGTH, MASS, TEMPERATURE, TIME

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend

Mass = typing.Annotated[Quantity, MASS]
MassOverADomain = typing.Annotated[Tensor, MASS]
Force = typing.Annotated[Quantity, FORCE]
Acceleration = typing.Annotated[Quantity, LENGTH / TIME**2]
AbsoluteTemperature = typing.Annotated[Quantity, TEMPERATURE, ABSOLUTE]
co2 = Metadata("co2")
TaggedMass = typing.Annotated[Quantity, MASS * co2]
time = dim("time")


@system
class Scale(metaclass=Interface):
    m: typing.Annotated[Quantity | QuantityTensor, MASS] = field()
    k: typing.Annotated[Quantity | QuantityTensor, MASS] = field()
    doubled: typing.Annotated[Quantity | QuantityTensor, MASS] = field()
    total: typing.Annotated[Quantity | QuantityTensor, MASS] = field()


@equation
def doubling(doubled: Mass, m: Mass) -> bool:
    return doubled == m * 2.0


@equation
def totalling(total: Mass, m: Mass, k: Mass) -> bool:
    return total == m + k


@system
class TaggedScale(metaclass=Interface):
    m: typing.Annotated[Quantity | QuantityTensor, MASS * co2] = field()
    doubled: typing.Annotated[Quantity | QuantityTensor, MASS * co2] = field()


@equation
def tagged_doubling(doubled: TaggedMass, m: TaggedMass) -> bool:
    return doubled == m * 2.0


@system
class TensorScale(metaclass=Interface):
    m: MassOverADomain = field()
    doubled: typing.Annotated[Quantity | QuantityTensor, MASS] = field()


@equation
def tensor_doubling(doubled: Mass, m: Mass) -> bool:
    return doubled == m * 2.0


@system
class PlainScale(metaclass=Interface):
    x: Tensor = field()
    doubled: Tensor = field()


@equation
def plain_doubling(doubled: Tensor, x: Tensor) -> bool:
    return doubled == x * 2.0


@system
class AbsoluteThermometer(metaclass=Interface):
    t: typing.Annotated[Quantity | QuantityTensor, TEMPERATURE, ABSOLUTE] = field()
    same: typing.Annotated[Quantity | QuantityTensor, TEMPERATURE, ABSOLUTE] = field()


@equation
def same_temperature(same: AbsoluteTemperature, t: AbsoluteTemperature) -> bool:
    return same == t


def _doubled(m: QuantityTensor) -> object:
    with Implementation(Scale, [doubling]):
        return Scale(m=m).doubled


def test_a_declared_answer_over_a_domain_comes_back_in_the_coherent_unit() -> None:
    result = _doubled(tensor([1.0, 2.0], over=time, unit=tonne))

    assert isinstance(result, Tensor)
    assert result.unit is kilogram
    np.testing.assert_allclose(result.values, [2000.0, 4000.0])


def test_a_declared_answer_over_a_domain_converts() -> None:
    result = _doubled(tensor([1.0, 2.0], over=time, unit=tonne))

    assert isinstance(result, Tensor)
    np.testing.assert_allclose(result.convert_to(tonne).values, [2.0, 4.0])


def test_an_answer_fed_back_in_as_an_input_is_read_in_its_own_unit() -> None:
    with Implementation(Scale, [doubling]):
        scale = Scale(m=tensor([1.0, 2.0], over=time, unit=tonne))
        first = scale.doubled
        second = scale.doubled

    assert isinstance(first, Tensor)
    assert isinstance(second, Tensor)
    np.testing.assert_allclose(second.values, first.values)


def test_a_tensor_and_a_scalar_quantity_may_meet_in_one_equation() -> None:
    masses = tensor([1.0, 2.0], over=time, unit=kilogram)
    with Implementation(Scale, [totalling]):
        result = Scale(m=masses, k=Quantity(500.0, gram)).total

    assert isinstance(result, Tensor)
    assert result.unit is kilogram
    np.testing.assert_allclose(result.values, [1.5, 2.5])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_a_unit_carrying_tensor_is_checked_and_labelled_on_either_backend(backend: Backend) -> None:
    with backend:
        result = _doubled(tensor([1.0, 2.0], over=time, unit=tonne))

    assert isinstance(result, Tensor)
    assert result.unit is kilogram
    np.testing.assert_allclose(result.values, [2000.0, 4000.0])


def test_a_scalar_declared_answer_is_a_quantity_holding_a_float() -> None:
    @transform
    def total(m: MassOverADomain) -> Mass:
        return m.sum(time)  # pyrefly: ignore[bad-return]  # ty: ignore[invalid-return-type]

    with Implementation(Scale, [total]):
        result = Scale(m=tensor([1.0, 2.0], over=time, unit=kilogram)).total

    assert isinstance(result, Quantity)
    assert type(result.value) is float
    assert result == 3.0 * kilogram


def test_a_transform_bridges_a_plain_tensor_into_a_declared_port() -> None:
    @system
    class Rig(metaclass=Interface):
        masses_g: Tensor = field()
        a: typing.Annotated[Quantity | QuantityTensor, LENGTH / TIME**2] = field()
        force: typing.Annotated[Quantity | QuantityTensor, FORCE] = field()

    @transform
    def force(masses_g: Tensor, a: Acceleration) -> Force:
        return masses_g * gram * a  # pyrefly: ignore[bad-return]  # ty: ignore[invalid-return-type]

    with Implementation(Rig, [force]):
        result = Rig(masses_g=tensor([1000.0, 2000.0], over=time), a=2.0 * meter / second**2).force

    assert isinstance(result, Tensor)
    assert result.unit is kilogram * meter / second**2
    np.testing.assert_allclose(result.values, [2.0, 4.0])


def test_a_declared_tensor_parameter_is_erased_in_a_transform_body_as_a_scalar_is() -> None:
    @transform
    def doubled(m: Mass) -> Mass:
        return Quantity(m.convert_to(gram).value * 2.0, gram)

    with Implementation(Scale, [doubled]):
        result = Scale(m=tensor([1.0, 2.0], over=time, unit=kilogram)).doubled

    assert isinstance(result, Tensor)
    assert result.unit is kilogram
    np.testing.assert_allclose(result.values, [2.0, 4.0])


def test_an_absolute_declared_tensor_parameter_is_erased_with_its_offset() -> None:
    @system
    class Rig(metaclass=Interface):
        t: typing.Annotated[Quantity | QuantityTensor, TEMPERATURE, ABSOLUTE] = field()
        in_celsius: typing.Annotated[Quantity | QuantityTensor, TEMPERATURE, ABSOLUTE] = field()

    @transform
    def in_celsius(t: AbsoluteTemperature) -> AbsoluteTemperature:
        return Quantity(t.convert_to(absolute_celsius).value, absolute_celsius)

    temperatures = tensor([273.15, 373.15], over=time, unit=abs(kelvin))
    with Implementation(Rig, [in_celsius]):
        result = Rig(t=temperatures).in_celsius

    assert isinstance(result, Tensor)
    assert result.unit is abs(kelvin)
    np.testing.assert_allclose(result.values, [273.15, 373.15])


def test_a_tagged_tensor_answers_in_its_tagged_coherent_unit() -> None:
    masses = tensor([1.0, 2.0], over=time, unit=kilogram * co2)
    with Implementation(TaggedScale, [tagged_doubling]):
        result = TaggedScale(m=masses).doubled

    assert isinstance(result, Tensor)
    assert result.unit == kilogram * co2
    np.testing.assert_allclose(result.values, [2.0, 4.0])


def test_an_untagged_tensor_at_a_tagged_port_raises() -> None:
    with pytest.raises(BiltInputDimensionalityError, match="'m'"):
        TaggedScale(m=tensor([1.0, 2.0], over=time, unit=kilogram))


def test_stripping_a_tag_in_a_transform_body_passes_a_tensor_through() -> None:
    @system
    class Rig(metaclass=Interface):
        m: typing.Annotated[Quantity | QuantityTensor, MASS * co2] = field()
        plain: typing.Annotated[Quantity | QuantityTensor, MASS] = field()

    @transform
    def plain(m: TaggedMass) -> Mass:
        return m.strip_metadata()

    masses = tensor([1.0, 2.0], over=time, unit=kilogram * co2)
    with Implementation(Rig, [plain]):
        result = Rig(m=masses).plain

    assert isinstance(result, Tensor)
    assert result.unit is kilogram
    np.testing.assert_allclose(result.values, [1.0, 2.0])


def test_multiplying_by_a_tag_in_a_transform_body_passes_a_tensor_through() -> None:
    @system
    class Rig(metaclass=Interface):
        m: typing.Annotated[Quantity | QuantityTensor, MASS] = field()
        tagged: typing.Annotated[Quantity | QuantityTensor, MASS * co2] = field()

    @transform
    def tagged(m: Mass) -> TaggedMass:
        return m * co2

    masses = tensor([1.0, 2.0], over=time, unit=kilogram)
    with Implementation(Rig, [tagged]):
        result = Rig(m=masses).tagged

    assert isinstance(result, Tensor)
    assert result.unit == kilogram * co2
    np.testing.assert_allclose(result.values, [1.0, 2.0])


def test_one_port_may_be_a_tensor_on_the_interface_and_a_quantity_in_a_constituent() -> None:
    with Implementation(TensorScale, [tensor_doubling]):
        result = TensorScale(m=tensor([1.0, 2.0], over=time, unit=kilogram)).doubled

    assert isinstance(result, Tensor)
    np.testing.assert_allclose(result.values, [2.0, 4.0])


def test_a_rank_zero_tensor_at_a_tensor_annotated_port_is_a_tensor() -> None:
    with Implementation(TensorScale, [tensor_doubling]):
        result = TensorScale(m=tensor(1.0, unit=kilogram)).doubled

    assert result == 2.0 * kilogram


def test_a_tensor_with_no_unit_at_a_plain_port_stays_bare() -> None:
    with Implementation(PlainScale, [plain_doubling]):
        result = PlainScale(x=tensor([1.0, 2.0], over=time)).doubled

    assert isinstance(result, Tensor)
    assert result.unit is None
    np.testing.assert_allclose(result.values, [2.0, 4.0])


def test_an_absolute_tensor_answers_in_the_absolute_coherent_unit() -> None:
    temperatures = tensor([0.0, 100.0], over=time, unit=absolute_celsius)
    with Implementation(AbsoluteThermometer, [same_temperature]):
        result = AbsoluteThermometer(t=temperatures).same

    assert isinstance(result, Tensor)
    assert result.unit is abs(kelvin)
    np.testing.assert_allclose(result.values, [273.15, 373.15])


def test_a_dimensionality_differing_under_one_port_name_is_still_a_conflict() -> None:
    @system
    class Rig(metaclass=Interface):
        m: MassOverADomain = field()
        doubled: typing.Annotated[Tensor, FORCE] = field()

    @equation
    def doubling(doubled: Mass, m: Mass) -> bool:
        return doubled == m * 2.0

    with pytest.raises(BiltImplementationDimensionalityError, match="'doubled'"):
        Implementation(Rig, [doubling])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_rank_zero_tensor_outputs_can_be_reused(backend: Backend) -> None:
    @system
    class Chain(metaclass=Interface):
        x: MassOverADomain = field()
        y: MassOverADomain = field()
        z: MassOverADomain = field()

    @equation
    def twice(y: Mass, x: Mass) -> bool:
        return y == x * 2.0

    @equation
    def thrice(z: Mass, y: Mass) -> bool:
        return z == y * 3.0

    with backend, Implementation(Chain, [twice, thrice]):
        direct = Chain(x=tensor(1.0, unit=kilogram)).z

        staged = Chain(x=tensor(1.0, unit=kilogram))
        intermediate = staged.y
        sequential = staged.z

    assert isinstance(intermediate, Tensor)
    assert isinstance(direct, Tensor)
    assert isinstance(sequential, Tensor)
    assert intermediate.domain.is_scalar
    assert direct.domain.is_scalar
    assert sequential.domain.is_scalar
    assert intermediate.unit is kilogram
    assert sequential.unit is kilogram

    np.testing.assert_allclose(intermediate.values, 2.0)
    np.testing.assert_allclose(direct.values, 6.0)
    np.testing.assert_allclose(sequential.values, direct.values)


type FlexibleMass = typing.Annotated[Quantity | QuantityTensor, MASS]
type ScalarContractMass = typing.Annotated[Quantity, MASS]
type TensorContractMass = typing.Annotated[QuantityTensor, MASS]


@system
class GenericMassScale(metaclass=Interface):
    m: FlexibleMass = field()
    doubled: FlexibleMass = field()


@equation
def generic_mass_copy(m: ScalarContractMass, doubled: ScalarContractMass) -> bool:
    return doubled == m


def test_pointwise_quantity_boundary_accepts_scalar_and_tensor_without_ignores() -> None:
    mass = Quantity(2.0, kilogram)
    varied = tensor([1.0, 2.0], over=time, unit=kilogram)
    typing.assert_type(varied, QuantityTensor)
    with Implementation(GenericMassScale, [generic_mass_copy]):
        assert GenericMassScale(m=mass).doubled == mass
        answer = GenericMassScale(m=varied).doubled
    assert isinstance(answer, Tensor)
    np.testing.assert_array_equal(answer.values, [1.0, 2.0])


def test_scalar_and_tensor_quantity_boundaries_reject_wrong_carrier() -> None:
    @system
    class ScalarOnly(metaclass=Interface):
        mass: ScalarContractMass = field()

    @system
    class TensorOnly(metaclass=Interface):
        mass: TensorContractMass = field()

    with pytest.raises(BiltInputDimensionalityError):
        ScalarOnly(mass=typing.cast("Quantity", tensor([1.0], over=time, unit=kilogram)))
    with pytest.raises(BiltInputDimensionalityError):
        TensorOnly(mass=typing.cast("QuantityTensor", Quantity(2.0, kilogram)))
