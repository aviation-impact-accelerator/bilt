import typing

import pytest

from bilt import Interface, QuantityLike, dim, equation, field, system, transform
from bilt.core._port_declaration import PortDeclaration
from bilt.core._tensors._tensors import QuantityTensor, Tensor
from bilt.errors import BiltEquationDimensionalityError, BiltPortDeclarationError
from bilt.symbolic._domains import Domain
from bilt.units import Metadata, Quantity, kilogram
from bilt.units.dimensions import ABSOLUTE, AREA, DIMENSIONLESS, LENGTH, MASS, TEMPERATURE, TIME

if typing.TYPE_CHECKING:
    import collections.abc

Mass = typing.Annotated[Quantity, MASS]
MassField = typing.Annotated[QuantityLike, MASS]
Length = typing.Annotated[Quantity, LENGTH]
Dimensionless = typing.Annotated[Quantity, DIMENSIONLESS]
Kilograms = typing.Annotated[Quantity, kilogram]
MassAsFloat = typing.Annotated[float, MASS]
MassAndLength = typing.Annotated[Quantity, MASS, LENGTH]
type MassAlias = typing.Annotated[Quantity, MASS]
type MassAliasField = typing.Annotated[QuantityLike, MASS]
type MassAliasAlias = MassAlias
type MassAliasAliasField = MassAliasField
Temperature = typing.Annotated[Quantity, TEMPERATURE]
TemperatureField = typing.Annotated[QuantityLike, TEMPERATURE]
AbsoluteTemperature = typing.Annotated[Quantity, TEMPERATURE, ABSOLUTE]
AbsoluteTemperatureField = typing.Annotated[QuantityLike, TEMPERATURE, ABSOLUTE]
AbsoluteTime = typing.Annotated[Quantity, TIME, ABSOLUTE]
AbsoluteTemperatureRate = typing.Annotated[Quantity, TEMPERATURE / TIME, ABSOLUTE]
AbsoluteFloat = typing.Annotated[float, ABSOLUTE]
TaggedMass = typing.Annotated[Quantity, MASS * Metadata("co2")]
TaggedMassField = typing.Annotated[QuantityLike, MASS * Metadata("co2")]
MassBesideTag = typing.Annotated[Quantity, MASS, Metadata("co2")]
MassOverADomain = typing.Annotated[QuantityTensor, MASS]
MassOverADomainField = typing.Annotated[QuantityLike[Tensor], MASS]
AbsoluteTemperatureOverADomain = typing.Annotated[QuantityTensor, TEMPERATURE, ABSOLUTE]
AbsoluteTemperatureOverADomainField = typing.Annotated[QuantityLike[Tensor], TEMPERATURE, ABSOLUTE]
TaggedMassOverADomain = typing.Annotated[QuantityTensor, MASS * Metadata("co2")]
TaggedMassOverADomainField = typing.Annotated[QuantityLike[Tensor], MASS * Metadata("co2")]
KilogramsOverADomain = typing.Annotated[QuantityTensor, kilogram]
MassAndLengthOverADomain = typing.Annotated[QuantityTensor, MASS, LENGTH]


def test_system_records_only_the_declared_fields() -> None:
    @system
    class Rig(metaclass=Interface):
        plain: float = field()
        mass: MassField = field()

    assert Rig.__bilt_port_declarations__ == {"mass": PortDeclaration(MASS)}


def test_equation_records_each_declared_parameter() -> None:
    @equation
    def balance(m: Mass, length: Length, area: typing.Annotated[Quantity, AREA]) -> bool:
        return area == length * length + m * 0

    assert balance.__bilt_port_declarations__ == {
        "m": PortDeclaration(MASS),
        "length": PortDeclaration(LENGTH),
        "area": PortDeclaration(AREA),
    }


def test_equation_records_the_child_dimensionality_of_each_referenced_sub_system_port() -> None:
    @system
    class Child(metaclass=Interface):
        v: MassField = field()
        unused: float = field()

    @equation
    def link(child: Child, x: Mass) -> bool:
        return child.v == x

    assert link.__bilt_port_declarations__ == {"child.v": PortDeclaration(MASS), "x": PortDeclaration(MASS)}


def test_transform_records_its_return_annotation_under_its_own_name() -> None:
    @transform
    def doubled(m: Mass, k: Length) -> Mass:
        return m * k

    assert doubled.__bilt_port_declarations__ == {
        "m": PortDeclaration(MASS),
        "k": PortDeclaration(LENGTH),
        "doubled": PortDeclaration(MASS),
    }


def test_transform_without_a_return_annotation_does_not_declare_its_output() -> None:
    @transform
    def doubled(m):  # noqa: ANN001, ANN202
        return m * 2.0

    assert doubled.__bilt_port_declarations__ == {}


def test_quoted_annotations_are_resolved() -> None:
    @system
    class Rig(metaclass=Interface):
        mass: "MassField" = field()  # noqa: UP037
        limit: "typing.ClassVar[float]" = 1.0  # noqa: UP037

    @equation
    def balance(m: "Mass", n: "Mass") -> bool:  # noqa: UP037
        return m == n

    @transform
    def doubled(m: Mass) -> "Mass":  # noqa: UP037
        return m * 2.0

    assert Rig.__bilt_port_declarations__ == {"mass": PortDeclaration(MASS)}
    assert balance.__bilt_port_declarations__ == {"m": PortDeclaration(MASS), "n": PortDeclaration(MASS)}
    assert doubled.__bilt_port_declarations__ == {"m": PortDeclaration(MASS), "doubled": PortDeclaration(MASS)}


def test_type_statement_aliases_are_resolved() -> None:
    @system
    class Rig(metaclass=Interface):
        mass: MassAliasField = field()
        nested: MassAliasAliasField = field()

    @equation
    def balance(m: MassAlias, n: MassAliasAlias) -> bool:
        return m == n

    @transform
    def doubled(m: Mass) -> MassAlias:
        return m * 2.0

    assert Rig.__bilt_port_declarations__ == {"mass": PortDeclaration(MASS), "nested": PortDeclaration(MASS)}
    assert balance.__bilt_port_declarations__ == {"m": PortDeclaration(MASS), "n": PortDeclaration(MASS)}
    assert doubled.__bilt_port_declarations__ == {"m": PortDeclaration(MASS), "doubled": PortDeclaration(MASS)}


def test_foreign_metadata_is_ignored() -> None:
    @system
    class Rig(metaclass=Interface):
        noted_mass: typing.Annotated[QuantityLike, "note", MASS] = field()
        noted_float: typing.Annotated[float, "note"] = field()

    assert Rig.__bilt_port_declarations__ == {"noted_mass": PortDeclaration(MASS)}


def test_a_tagged_dimensionality_is_recorded() -> None:
    @system
    class Rig(metaclass=Interface):
        emitted: TaggedMassField = field()

    assert Rig.__bilt_port_declarations__ == {"emitted": PortDeclaration(MASS * Metadata("co2"))}


def test_an_absolute_marker_is_recorded_beside_the_dimensionality() -> None:
    @system
    class Engine(metaclass=Interface):
        hot: AbsoluteTemperatureField = field()
        drop: TemperatureField = field()

    @equation
    def cooling(hot: AbsoluteTemperature, cold: AbsoluteTemperature, drop: Temperature) -> bool:
        return drop == hot - cold

    @transform
    def cold(hot: AbsoluteTemperature, drop: Temperature) -> AbsoluteTemperature:
        return hot - drop

    absolute, relative = PortDeclaration(TEMPERATURE, absolute=True), PortDeclaration(TEMPERATURE)
    assert Engine.__bilt_port_declarations__ == {"hot": absolute, "drop": relative}
    assert cooling.__bilt_port_declarations__ == {"hot": absolute, "cold": absolute, "drop": relative}
    assert cold.__bilt_port_declarations__ == {"hot": absolute, "drop": relative, "cold": absolute}


def test_transform_may_mix_declared_and_plain_ports() -> None:
    @transform
    def doubled(m: Mass, k: float, flag: bool) -> Mass:  # noqa: ARG001
        return m * k

    assert doubled.__bilt_port_declarations__ == {"m": PortDeclaration(MASS), "doubled": PortDeclaration(MASS)}


def test_equation_mixing_declared_and_plain_ports_in_a_product_raises() -> None:
    def balance(m: Mass, k: float, n: Mass) -> bool:
        return m == k * n

    with pytest.raises(
        BiltEquationDimensionalityError,
        match=r"Equation 'balance' mixes declared and plain ports in one operation: plain 'k' with declared 'n'",
    ):
        equation(balance)


def test_equation_mixing_declared_and_plain_ports_in_a_sum_raises() -> None:
    def balance(m: Mass, k: float, n: Mass) -> bool:
        return m == n + k  # pyrefly: ignore[unsupported-operation]  # ty: ignore[unsupported-operator]

    with pytest.raises(
        BiltEquationDimensionalityError,
        match=r"Equation 'balance' mixes declared and plain ports in one operation: plain 'k' with declared 'n'",
    ):
        equation(balance)


def test_equation_relating_a_plain_port_to_a_declared_one_raises() -> None:
    def balance(m: Mass, k: float) -> bool:
        return k == m

    with pytest.raises(
        BiltEquationDimensionalityError,
        match=r"Equation 'balance' mixes declared and plain ports in one operation: plain 'k' with declared 'm'",
    ):
        equation(balance)


def test_equation_whose_sub_system_declares_ports_cannot_relate_them_to_a_plain_port() -> None:
    @system
    class Child(metaclass=Interface):
        v: MassField = field()

    def link(child: Child, x: float) -> bool:
        return child.v == x

    with pytest.raises(BiltEquationDimensionalityError, match=r"plain 'x' with declared 'child\.v'"):
        equation(link)


def test_equation_whose_sub_system_leaves_ports_plain_may_leave_its_own_plain() -> None:
    @system
    class Child(metaclass=Interface):
        v: float = field()

    @equation
    def link(child: Child, x: float) -> bool:
        return child.v == x

    assert link.__bilt_port_declarations__ == {}


def _system_field(annotation: object) -> None:
    namespace = {"__annotations__": {"m": annotation}, "m": field()}
    system(Interface("Rig", (), namespace))


def _equation_parameter(annotation: object) -> None:
    def rig(m, k: float) -> bool:  # noqa: ANN001
        return m == k

    rig.__annotations__["m"] = annotation
    equation(rig)


def _transform_parameter(annotation: object) -> None:
    def rig(m, k: float) -> float:  # noqa: ANN001
        return m * k

    rig.__annotations__["m"] = annotation
    transform(rig)


def _transform_return(annotation: object) -> None:
    def rig(k: float):  # noqa: ANN202
        return k

    rig.__annotations__["return"] = annotation
    transform(rig)


SITES = (
    pytest.param(_system_field, "Rig", "m", id="system field"),
    pytest.param(_equation_parameter, "rig", "m", id="equation parameter"),
    pytest.param(_transform_parameter, "rig", "m", id="transform parameter"),
    pytest.param(_transform_return, "rig", "rig", id="transform return"),
)


@pytest.mark.parametrize(("declare", "owner", "port"), SITES)
def test_a_bare_quantity_raises_and_points_both_ways_forward(
    declare: collections.abc.Callable[[object], None],
    owner: str,
    port: str,
) -> None:
    with pytest.raises(BiltPortDeclarationError, match=f"Port '{port}' of '{owner}' is a Quantity without") as raised:
        declare(Quantity)

    message = str(raised.value)
    assert "Annotated[Quantity, DIM]" in message
    assert "write `float`" in message


@pytest.mark.parametrize(("declare", "owner", "port"), SITES)
def test_dimensionality_on_a_non_quantity_base_raises(
    declare: collections.abc.Callable[[object], None],
    owner: str,
    port: str,
) -> None:
    with pytest.raises(BiltPortDeclarationError, match=f"Port '{port}' of '{owner}'") as raised:
        declare(MassAsFloat)

    message = str(raised.value)
    assert "mass" in message
    assert "float" in message


@pytest.mark.parametrize(("declare", "owner", "port"), SITES)
def test_a_unit_on_a_tensor_raises_as_it_does_on_a_quantity(
    declare: collections.abc.Callable[[object], None],
    owner: str,
    port: str,
) -> None:
    with pytest.raises(BiltPortDeclarationError, match=f"Port '{port}' of '{owner}'") as raised:
        declare(KilogramsOverADomain)

    message = str(raised.value)
    assert "kilogram" in message
    assert "mass" in message


@pytest.mark.parametrize(("declare", "owner", "port"), SITES)
def test_several_dimensionalities_on_a_tensor_raise_as_they_do_on_a_quantity(
    declare: collections.abc.Callable[[object], None],
    owner: str,
    port: str,
) -> None:
    with pytest.raises(BiltPortDeclarationError, match=f"Port '{port}' of '{owner}'") as raised:
        declare(MassAndLengthOverADomain)

    assert "more than one dimensionality" in str(raised.value)


@pytest.mark.parametrize(("declare", "owner", "port"), SITES)
def test_several_dimensionalities_raise(
    declare: collections.abc.Callable[[object], None],
    owner: str,
    port: str,
) -> None:
    with pytest.raises(BiltPortDeclarationError, match=f"Port '{port}' of '{owner}'") as raised:
        declare(MassAndLength)

    message = str(raised.value)
    assert "mass" in message
    assert "length" in message


@pytest.mark.parametrize(("declare", "owner", "port"), SITES)
def test_unit_raises(
    declare: collections.abc.Callable[[object], None],
    owner: str,
    port: str,
) -> None:
    with pytest.raises(BiltPortDeclarationError, match=f"Port '{port}' of '{owner}'") as raised:
        declare(Kilograms)

    message = str(raised.value)
    assert "kilogram" in message
    assert "mass" in message


@pytest.mark.parametrize(("declare", "owner", "port"), SITES)
def test_a_tag_beside_the_dimensionality_raises(
    declare: collections.abc.Callable[[object], None],
    owner: str,
    port: str,
) -> None:
    with pytest.raises(BiltPortDeclarationError, match=f"Port '{port}' of '{owner}' declares tag co2") as raised:
        declare(MassBesideTag)

    assert "DIM * tag" in str(raised.value)


@pytest.mark.parametrize(("declare", "owner", "port"), SITES)
@pytest.mark.parametrize("annotation", (AbsoluteTime, AbsoluteTemperatureRate))
def test_an_absolute_marker_where_the_coherent_unit_has_no_absolute_counterpart_raises(
    declare: collections.abc.Callable[[object], None],
    owner: str,
    port: str,
    annotation: object,
) -> None:
    if declare is _system_field:
        annotation = (
            typing.Annotated[QuantityLike, TIME, ABSOLUTE]
            if annotation == AbsoluteTime
            else typing.Annotated[QuantityLike, TEMPERATURE / TIME, ABSOLUTE]
        )

    with pytest.raises(BiltPortDeclarationError, match=f"Port '{port}' of '{owner}' is marked ABSOLUTE") as raised:
        declare(annotation)

    assert "cannot be absolute" in str(raised.value)


@pytest.mark.parametrize(("declare", "owner", "port"), SITES)
def test_an_absolute_marker_without_a_dimensionality_raises(
    declare: collections.abc.Callable[[object], None],
    owner: str,
    port: str,
) -> None:
    with pytest.raises(BiltPortDeclarationError, match=f"Port '{port}' of '{owner}' is marked ABSOLUTE without"):
        declare(AbsoluteFloat)


def test_a_dimensionality_on_a_tensor_is_the_same_declaration_as_on_a_quantity() -> None:
    @system
    class Rig(metaclass=Interface):
        over_a_domain: MassOverADomainField = field()
        scalar: MassField = field()

    assert Rig.__bilt_port_declarations__ == {
        "over_a_domain": PortDeclaration(MASS),
        "scalar": PortDeclaration(MASS),
    }


def test_a_tensor_port_records_that_it_varies_over_a_domain() -> None:
    @system
    class Rig(metaclass=Interface):
        over_a_domain: MassOverADomainField = field()
        scalar: MassField = field()

    assert Rig.__bilt_port_declarations__["over_a_domain"].over_a_domain
    assert not Rig.__bilt_port_declarations__["scalar"].over_a_domain


def test_varying_over_a_domain_is_not_part_of_a_declaration_s_identity() -> None:
    assert PortDeclaration(MASS, over_a_domain=True) == PortDeclaration(MASS, over_a_domain=False)


def test_a_bare_tensor_is_still_a_plain_port() -> None:
    @system
    class Rig(metaclass=Interface):
        m: Tensor = field()

    assert Rig.__bilt_port_declarations__ == {}


def test_the_absolute_marker_reads_on_a_tensor() -> None:
    @system
    class Rig(metaclass=Interface):
        t: AbsoluteTemperatureOverADomainField = field()

    assert Rig.__bilt_port_declarations__ == {"t": PortDeclaration(TEMPERATURE, absolute=True)}


def test_a_tag_reads_on_a_tensor() -> None:
    @system
    class Rig(metaclass=Interface):
        m: TaggedMassOverADomainField = field()

    assert Rig.__bilt_port_declarations__ == {"m": PortDeclaration(MASS * Metadata("co2"))}


def test_a_tensor_annotation_reads_at_an_equation_and_a_transform() -> None:
    @equation
    def balance(m: MassOverADomain, n: Mass) -> bool:
        return m == n

    @transform
    def doubled(m: MassOverADomain) -> MassOverADomain:
        return m * 2.0  # pyrefly: ignore[bad-return]  # ty: ignore[invalid-return-type]

    assert balance.__bilt_port_declarations__ == {"m": PortDeclaration(MASS), "n": PortDeclaration(MASS)}
    assert doubled.__bilt_port_declarations__ == {"m": PortDeclaration(MASS), "doubled": PortDeclaration(MASS)}


def test_generic_quantity_annotation_distinguishes_scalar_and_tensor_ports() -> None:
    @system
    class Variants(metaclass=Interface):
        flexible: typing.Annotated[QuantityLike, MASS] = field()
        scalar: typing.Annotated[QuantityLike[float], MASS] = field()
        tensor: typing.Annotated[QuantityLike[Tensor], MASS] = field()
        concrete: typing.Annotated[QuantityTensor, MASS] = field()

    declarations = Variants.__bilt_port_declarations__
    assert not declarations["flexible"].scalar_only
    assert declarations["scalar"].scalar_only
    assert declarations["tensor"].over_a_domain
    assert declarations["concrete"].over_a_domain
    assert all(declaration == PortDeclaration(MASS) for declaration in declarations.values())


def test_bare_generic_quantity_still_requires_physical_dimension() -> None:
    with pytest.raises(BiltPortDeclarationError, match="without a dimensionality"):

        @system
        class MissingDimension(metaclass=Interface):
            mass: QuantityLike = field()


def test_new_quantity_tensor_domain_declares_minimum_named_axes() -> None:
    time = dim("time")

    @system
    class Rig(metaclass=Interface):
        mass: typing.Annotated[QuantityLike[Tensor], MASS, Domain((time,))] = field()
        scalar: typing.Annotated[QuantityLike, MASS] = field()

    declared = Rig.__bilt_port_declarations__["mass"]
    assert declared.required_domain == Domain((time,))
    assert declared.over_a_domain
    assert declared == Rig.__bilt_port_declarations__["scalar"]


def test_minimum_domain_metadata_rejects_duplicate_and_non_tensor_contracts() -> None:
    time = dim("time")
    scenario = dim("scenario")
    with pytest.raises(BiltPortDeclarationError, match="more than one required domain"):

        @system
        class Duplicate(metaclass=Interface):
            mass: typing.Annotated[QuantityTensor, MASS, Domain((time,)), Domain((scenario,))] = field()

    with pytest.raises(BiltPortDeclarationError, match="scalar or generic quantity"):

        @system
        class Scalar(metaclass=Interface):
            mass: typing.Annotated[QuantityLike[float], MASS, Domain((time,))] = field()

    with pytest.raises(BiltPortDeclarationError, match="scalar or generic quantity"):

        @system
        class Generic(metaclass=Interface):
            mass: typing.Annotated[QuantityLike, MASS, Domain((time,))] = field()


def test_unitless_tensor_domain_hints_do_not_declare_physical_units() -> None:
    time = dim("time")

    @system
    class Plain(metaclass=Interface):
        unitless: typing.Annotated[Tensor, Domain((time,))] = field()

    assert Plain.__bilt_port_declarations__ == {}


def test_dimensioned_quantity_system_field_uses_new_api() -> None:
    with pytest.raises(BiltPortDeclarationError, match="old Quantity system-field annotation"):
        _system_field(typing.Annotated[Quantity, MASS])


@pytest.mark.parametrize("declare", (_system_field, _equation_parameter, _transform_parameter, _transform_return))
def test_dimensioned_tensor_annotation_uses_new_api(declare: collections.abc.Callable[[object], None]) -> None:
    with pytest.raises(BiltPortDeclarationError, match="old dimensioned Tensor annotation"):
        declare(typing.Annotated[Tensor, MASS])
