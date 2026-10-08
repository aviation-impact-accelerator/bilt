"""A tag is a factor of the dimensionality a port declares: demanded on the way in and restored on the way out."""

import typing

import pytest

from bilt import JAX_BACKEND, PYTHON_BACKEND, Implementation, Interface, equation, field, system, transform
from bilt.errors import (
    BiltDefinitionError,
    BiltEquationDimensionalityError,
    BiltImplementationDimensionalityError,
    BiltInputDimensionalityError,
    BiltPortDeclarationError,
    BiltSymbolicConversionError,
)
from bilt.units import Metadata, Quantity
from bilt.units.dimensions import ABSOLUTE, AMOUNT, DIMENSIONLESS, MASS, TEMPERATURE
from bilt.units.units import absolute_celsius, fraction, gram, kilogram, mole, percent

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend

co2 = Metadata("co2")
fuel = Metadata("fuel")
gas = Metadata("gas")
mass = Metadata("mass")
hydrogen = Metadata("hydrogen")
carbon_monoxide = Metadata("carbon_monoxide")
mass_fraction = fraction * mass

Mass = typing.Annotated[Quantity, MASS]
FuelMass = typing.Annotated[Quantity, MASS * fuel]
Co2Mass = typing.Annotated[Quantity, MASS * co2]
EmissionIndex = typing.Annotated[Quantity, (MASS * co2) / (MASS * fuel)]
MassFraction = typing.Annotated[Quantity, DIMENSIONLESS * mass]
Hydrogen = typing.Annotated[Quantity, AMOUNT * hydrogen]
CarbonMonoxide = typing.Annotated[Quantity, AMOUNT * carbon_monoxide]
SyngasRatio = typing.Annotated[Quantity, (AMOUNT * hydrogen) / (AMOUNT * carbon_monoxide)]
GasTemperature = typing.Annotated[Quantity, TEMPERATURE * gas]


@system
class Burn(metaclass=Interface):
    burned: FuelMass = field()
    index: EmissionIndex = field()
    emitted: Co2Mass = field()


@equation
def emission(emitted: Co2Mass, burned: FuelMass, index: EmissionIndex) -> bool:
    return emitted == burned * index


@system
class Split(metaclass=Interface):
    total: Mass = field()
    x: MassFraction = field()
    part: Mass = field()


@equation
def split(part: Mass, total: Mass, x: MassFraction) -> bool:
    return part == total * x / mass_fraction


@system
class Fixed(metaclass=Interface):
    burned: FuelMass = field()
    emitted: Co2Mass = field()


@system
class Weighed(metaclass=Interface):
    burned: FuelMass = field()
    grams: float = field()
    emitted: Co2Mass = field()


def test_a_tagged_port_answers_in_its_tagged_coherent_unit() -> None:
    with Implementation(Burn, [emission]):
        emitted = Burn(burned=2000.0 * gram * fuel, index=3.16 * (kilogram * co2) / (kilogram * fuel)).emitted

    assert emitted.value == pytest.approx(6.32)
    assert emitted.unit is kilogram * co2


def test_an_equation_is_solved_for_a_tagged_port_it_multiplies() -> None:
    with Implementation(Burn, [emission]):
        burned = Burn(emitted=6.32 * kilogram * co2, index=3160.0 * (gram * co2) / (kilogram * fuel)).burned

    assert burned.value == pytest.approx(2.0)
    assert burned.unit is kilogram * fuel


def test_a_quotient_of_tagged_ports_answers_with_each_tag() -> None:
    @system
    class Syngas(metaclass=Interface):
        h2: Hydrogen = field()
        co: CarbonMonoxide = field()
        ratio: SyngasRatio = field()

    @equation
    def syngas_ratio(ratio: SyngasRatio, h2: Hydrogen, co: CarbonMonoxide) -> bool:
        return ratio == h2 / co

    with Implementation(Syngas, [syngas_ratio]):
        ratio = Syngas(h2=4.0 * mole * hydrogen, co=2.0 * mole * carbon_monoxide).ratio

    assert ratio.value == pytest.approx(2.0)
    assert ratio.unit is (mole * hydrogen) / (mole * carbon_monoxide)


def test_a_tagged_fraction_is_answered_tagged() -> None:
    with Implementation(Split, [split]):
        x = Split(total=10.0 * kilogram, part=2500.0 * gram).x

    assert x.value == pytest.approx(0.25)
    assert x.unit == mass_fraction


def test_a_tagged_fraction_is_divided_out_to_scale_an_untagged_quantity() -> None:
    with Implementation(Split, [split]):
        part = Split(total=10.0 * kilogram, x=25.0 * percent * mass).part

    assert part.value == pytest.approx(2.5)
    assert part.unit is kilogram


@pytest.mark.parametrize(
    ("port", "quantity"),
    (
        pytest.param("burned", 2.0 * kilogram, id="untagged"),
        pytest.param("burned", 2.0 * kilogram * co2, id="another tag"),
        pytest.param("index", 3.16 * fraction, id="a compound without its tags"),
        pytest.param("index", 3.16 * (kilogram * fuel) / (kilogram * co2), id="a compound with its tags swapped"),
    ),
)
def test_a_quantity_without_the_declared_tags_is_refused(port: str, quantity: Quantity) -> None:
    with pytest.raises(BiltInputDimensionalityError, match=f"Input '{port}' of 'Burn'"):
        Burn(**{port: quantity})


def test_a_tagged_quantity_is_refused_at_an_untagged_port() -> None:
    with pytest.raises(BiltInputDimensionalityError, match="Input 'total' of 'Split'"):
        Split(total=10.0 * kilogram * fuel)


def test_an_absolute_quantity_with_a_tag_is_refused() -> None:
    @system
    class Flue(metaclass=Interface):
        temperature: GasTemperature = field()

    with pytest.raises(BiltInputDimensionalityError, match=r"Input 'temperature' of 'Flue'.*in a relative unit"):
        Flue(temperature=Quantity(20.0, absolute_celsius) * gas)


def test_absolute_on_a_tagged_dimensionality_raises() -> None:
    with pytest.raises(BiltPortDeclarationError, match="Port 'temperature' of 'Flue' is marked ABSOLUTE") as raised:

        @system
        class Flue(metaclass=Interface):
            temperature: typing.Annotated[Quantity, TEMPERATURE * gas, ABSOLUTE] = field()

    assert "cannot be absolute" in str(raised.value)


def test_a_bare_number_does_not_change_a_tag_in_an_equation() -> None:
    with pytest.raises(BiltEquationDimensionalityError, match="Equation 'fixed_emission'"):

        @equation
        def fixed_emission(emitted: Co2Mass, burned: FuelMass) -> bool:
            return emitted == burned * 3.16


def test_a_tagged_constant_changes_a_tag_in_an_equation() -> None:
    @equation
    def fixed_emission(emitted: Co2Mass, burned: FuelMass) -> bool:
        return emitted == burned * (3.16 * (kilogram * co2) / (kilogram * fuel))

    with Implementation(Fixed, [fixed_emission]):
        emitted = Fixed(burned=2000.0 * gram * fuel).emitted

    assert emitted.value == pytest.approx(6.32)
    assert emitted.unit is kilogram * co2


def test_a_tagged_fraction_left_in_a_product_is_refused() -> None:
    with pytest.raises(BiltEquationDimensionalityError, match="Equation 'unsplit'"):

        @equation
        def unsplit(part: Mass, total: Mass, x: MassFraction) -> bool:
            return part == total * x


def test_a_float_power_of_a_tagged_fraction_is_refused_in_an_equation() -> None:
    with pytest.raises(BiltEquationDimensionalityError, match="anything but an integer literal"):

        @equation
        def root(part: MassFraction, x: MassFraction) -> bool:
            return part == x**0.5


def test_a_port_declared_with_a_tag_in_one_place_and_without_in_another_raises() -> None:
    @equation
    def untagged_emission(emitted: Co2Mass, burned: Mass) -> bool:
        return emitted == burned * (3.16 * (kilogram * co2) / kilogram)

    with pytest.raises(BiltImplementationDimensionalityError, match="port 'burned' conflicting declarations"):
        Implementation(Fixed, [untagged_emission])


def test_a_tagged_constant_changes_a_tag_in_a_transform() -> None:
    @transform
    def emitted(burned: FuelMass) -> Co2Mass:
        return burned * (3.16 * (kilogram * co2) / (kilogram * fuel))

    with Implementation(Fixed, [emitted]):
        answer = Fixed(burned=2000.0 * gram * fuel).emitted

    assert answer.value == pytest.approx(6.32)
    assert answer.unit is kilogram * co2


def test_a_conversion_to_a_tagged_unit_takes_a_tagged_port_out_to_a_plain_one() -> None:
    @transform
    def grams(burned: FuelMass) -> float:
        return burned.convert_to(gram * fuel).value

    with Implementation(Weighed, [grams]):
        assert Weighed(burned=2.0 * kilogram * fuel).grams == pytest.approx(2000.0)


def test_a_constructed_tagged_quantity_brings_a_plain_port_into_a_tagged_one() -> None:
    @transform
    def emitted(grams: float) -> Co2Mass:
        return Quantity(grams, gram * co2)

    with Implementation(Weighed, [emitted]):
        answer = Weighed(grams=2000.0).emitted

    assert answer.value == pytest.approx(2.0)
    assert answer.unit is kilogram * co2


def test_converting_a_tagged_parameter_to_an_untagged_unit_is_refused() -> None:
    with pytest.raises(
        BiltDefinitionError,
        match=r"`burned\.convert_to\(kilogram\)` contradicts the signature",
    ):

        @transform
        def emitted(burned: FuelMass) -> Co2Mass:
            return burned.convert_to(kilogram) * (3.16 * (kilogram * co2) / kilogram)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_tags_are_shed_and_attached_in_a_transform(backend: Backend) -> None:
    @transform
    def emitted(burned: FuelMass) -> Co2Mass:
        return burned.strip_metadata() * 3.16 * co2

    with backend, Implementation(Fixed, [emitted]):
        answer = Fixed(burned=2000.0 * gram * fuel).emitted

    assert answer.value == pytest.approx(6.32)
    assert answer.unit is kilogram * co2


def test_a_tag_is_attached_to_a_local_in_a_transform() -> None:
    @transform
    def emitted(burned: FuelMass) -> Co2Mass:
        scaled = burned.strip_metadata() * 3.16
        return scaled * co2

    with Implementation(Fixed, [emitted]):
        answer = Fixed(burned=2000.0 * gram * fuel).emitted

    assert answer.value == pytest.approx(6.32)
    assert answer.unit is kilogram * co2


def test_a_tag_brings_a_plain_port_scaled_by_a_unit_into_a_tagged_one() -> None:
    @transform
    def emitted(grams: float) -> Co2Mass:
        return grams * gram * co2

    with Implementation(Weighed, [emitted]):
        answer = Weighed(grams=2000.0).emitted

    assert answer.value == pytest.approx(2.0)
    assert answer.unit is kilogram * co2


def test_a_tag_on_a_plain_parameter_is_not_erased() -> None:
    @transform
    def emitted(grams: float) -> Co2Mass:
        return grams * co2  # pyrefly: ignore[bad-return,unsupported-operation]  # ty: ignore[unsupported-operator]

    with Implementation(Weighed, [emitted]), pytest.raises(NameError, match="name 'co2' is not defined"):
        _ = Weighed(grams=2000.0).emitted


def test_a_tag_on_a_comprehension_variable_named_as_a_plain_parameter_is_erased() -> None:
    @transform
    def emitted(burned: FuelMass, grams: float) -> Co2Mass:
        tagged = [grams * co2 for grams in [burned.strip_metadata()]]
        return tagged[0] * 3.16 + grams * gram * co2

    with Implementation(Weighed, [emitted]):
        answer = Weighed(burned=2000.0 * gram * fuel, grams=0.0).emitted

    assert answer.value == pytest.approx(6.32)
    assert answer.unit is kilogram * co2


def test_a_cast_is_refused_with_the_spellings_that_relabel() -> None:
    with pytest.raises(
        BiltDefinitionError,
        match=r"`burned\.cast_to` depends on the unit `burned` arrived in: "
        r"shed its tags with `burned\.strip_metadata\(\)` or attach one with `burned \* <tag>`",
    ):

        @transform
        def emitted(burned: FuelMass) -> Co2Mass:
            return burned.cast_to(kilogram) * 3.16 * co2


def test_strip_metadata_given_an_argument_is_refused() -> None:
    with pytest.raises(
        BiltDefinitionError,
        match=r"`burned\.strip_metadata` must be called with no arguments: write `burned\.strip_metadata\(\)`",
    ):

        @transform
        def emitted(burned: FuelMass) -> Co2Mass:
            return burned.strip_metadata(kilogram) * 3.16 * co2  # pyrefly: ignore[bad-argument-count]  # ty: ignore[too-many-positional-arguments]


def test_a_tag_as_a_factor_is_refused_in_an_equation() -> None:
    with pytest.raises(
        BiltSymbolicConversionError,
        match=r"tag 'co2' as a factor in an equation.*`m \* \(kilogram \* co2\) / kilogram`",
    ):

        @equation
        def relabelled(emitted: Co2Mass, burned: Mass) -> bool:
            return emitted == burned * co2


def test_shedding_tags_is_refused_in_an_equation() -> None:
    with pytest.raises(BiltSymbolicConversionError, match=r"strip_metadata\(\)` in an equation.*`x / mass_fraction`"):

        @equation
        def unsplit(part: Mass, total: Mass, x: MassFraction) -> bool:
            return part == total * x.strip_metadata()
