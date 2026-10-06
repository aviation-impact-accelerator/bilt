import typing

import pytest

from bilt import Interface, equation, field, system, transform
from bilt.core._tensors._tensors import Tensor
from bilt.errors import BiltEquationDimensionalityError, BiltPortDeclarationError
from bilt.units import Quantity, kilogram
from bilt.units.dimensions import AREA, DIMENSIONLESS, LENGTH, MASS

if typing.TYPE_CHECKING:
    import collections.abc

Mass = typing.Annotated[Quantity, MASS]
Length = typing.Annotated[Quantity, LENGTH]
Dimensionless = typing.Annotated[Quantity, DIMENSIONLESS]
Kilograms = typing.Annotated[Quantity, kilogram]
MassAsFloat = typing.Annotated[float, MASS]
MassAsTensor = typing.Annotated[Tensor, MASS]
MassAndLength = typing.Annotated[Quantity, MASS, LENGTH]
type MassAlias = typing.Annotated[Quantity, MASS]
type MassAliasAlias = MassAlias


def test_system_records_only_the_declared_fields() -> None:
    @system
    class Rig(metaclass=Interface):
        plain: float = field()
        mass: Mass = field()

    assert Rig.__bilt_port_declarations__ == {"mass": MASS}


def test_a_wholly_plain_system_records_nothing() -> None:
    @system
    class Rig(metaclass=Interface):
        a: float = field()
        b = field()

    assert Rig.__bilt_port_declarations__ == {}


def test_system_interface_may_mix_declared_and_plain_fields() -> None:
    @system
    class Rig(metaclass=Interface):
        mass: Mass = field()
        plain: float = field()

    assert Rig.__bilt_port_declarations__ == {"mass": MASS}


def test_equation_records_each_declared_parameter() -> None:
    @equation
    def balance(m: Mass, length: Length, area: typing.Annotated[Quantity, AREA]) -> bool:
        return area == length * length + m * 0

    assert balance.__bilt_port_declarations__ == {"m": MASS, "length": LENGTH, "area": AREA}


def test_a_wholly_plain_equation_records_nothing() -> None:
    @equation
    def balance(m: float, k) -> bool:  # noqa: ANN001
        return m == m * k

    assert balance.__bilt_port_declarations__ == {}


def test_equation_records_the_child_dimensionality_of_each_referenced_sub_system_port() -> None:
    @system
    class Child(metaclass=Interface):
        v: Mass = field()
        unused: float = field()

    @equation
    def link(child: Child, x: Mass) -> bool:
        return child.v == x

    assert link.__bilt_port_declarations__ == {"child.v": MASS, "x": MASS}


def test_transform_records_its_return_annotation_under_its_own_name() -> None:
    @transform
    def doubled(m: Mass, k: Length) -> Mass:
        return m * k

    assert doubled.__bilt_port_declarations__ == {"m": MASS, "k": LENGTH, "doubled": MASS}


def test_a_wholly_plain_transform_records_nothing() -> None:
    @transform
    def doubled(m: float, k) -> float:  # noqa: ANN001
        return m * k

    assert doubled.__bilt_port_declarations__ == {}


def test_transform_without_a_return_annotation_does_not_declare_its_output() -> None:
    @transform
    def doubled(m):  # noqa: ANN001, ANN202
        return m * 2.0

    assert doubled.__bilt_port_declarations__ == {}


def test_transform_with_declared_parameters_and_no_return_annotation_leaves_its_output_plain() -> None:
    @transform
    def doubled(m: Mass):  # noqa: ANN202
        return m * 2.0

    assert doubled.__bilt_port_declarations__ == {"m": MASS}


def test_quoted_annotations_are_resolved() -> None:
    @system
    class Rig(metaclass=Interface):
        mass: "Mass" = field()  # noqa: UP037
        limit: "typing.ClassVar[float]" = 1.0  # noqa: UP037

    @equation
    def balance(m: "Mass", n: "Mass") -> bool:  # noqa: UP037
        return m == n

    @transform
    def doubled(m: Mass) -> "Mass":  # noqa: UP037
        return m * 2.0

    assert Rig.__bilt_port_declarations__ == {"mass": MASS}
    assert balance.__bilt_port_declarations__ == {"m": MASS, "n": MASS}
    assert doubled.__bilt_port_declarations__ == {"m": MASS, "doubled": MASS}


def test_type_statement_aliases_are_resolved() -> None:
    @system
    class Rig(metaclass=Interface):
        mass: MassAlias = field()
        nested: MassAliasAlias = field()

    @equation
    def balance(m: MassAlias, n: MassAliasAlias) -> bool:
        return m == n

    @transform
    def doubled(m: Mass) -> MassAlias:
        return m * 2.0

    assert Rig.__bilt_port_declarations__ == {"mass": MASS, "nested": MASS}
    assert balance.__bilt_port_declarations__ == {"m": MASS, "n": MASS}
    assert doubled.__bilt_port_declarations__ == {"m": MASS, "doubled": MASS}


def test_foreign_metadata_is_ignored() -> None:
    @system
    class Rig(metaclass=Interface):
        noted_mass: typing.Annotated[Quantity, "note", MASS] = field()
        noted_float: typing.Annotated[float, "note"] = field()

    assert Rig.__bilt_port_declarations__ == {"noted_mass": MASS}


def test_transform_may_mix_declared_and_plain_ports() -> None:
    @transform
    def doubled(m: Mass, k: float, flag: bool) -> Mass:  # noqa: ARG001
        return m * k

    assert doubled.__bilt_port_declarations__ == {"m": MASS, "doubled": MASS}


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


def test_equation_relating_a_plain_sum_to_a_declared_port_raises() -> None:
    def balance(total: float, a: Mass, b: Mass) -> bool:
        return total == a + b

    with pytest.raises(
        BiltEquationDimensionalityError,
        match=r"plain 'total' with declared 'a', 'b'",
    ):
        equation(balance)


def test_equation_using_a_plain_port_only_as_an_exponent_decorates() -> None:
    @equation
    def powered(y: Dimensionless, x: Dimensionless, n: float) -> bool:
        return y == x**n

    assert powered.__bilt_port_declarations__ == {"y": DIMENSIONLESS, "x": DIMENSIONLESS}


def test_equation_whose_sub_system_declares_ports_cannot_relate_them_to_a_plain_port() -> None:
    @system
    class Child(metaclass=Interface):
        v: Mass = field()

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
    def rig(m, n: Mass) -> bool:  # noqa: ANN001
        return m == n

    rig.__annotations__["m"] = annotation
    equation(rig)


def _transform_parameter(annotation: object) -> None:
    def rig(m) -> Mass:  # noqa: ANN001
        return m

    rig.__annotations__["m"] = annotation
    transform(rig)


def _transform_return(annotation: object) -> None:
    def rig(k: Mass):  # noqa: ANN202
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
def test_dimensionality_on_a_tensor_raises_with_the_scope_message(
    declare: collections.abc.Callable[[object], None],
    owner: str,
    port: str,
) -> None:
    with pytest.raises(BiltPortDeclarationError, match=f"Port '{port}' of '{owner}'") as raised:
        declare(MassAsTensor)

    assert "outside the unit system" in str(raised.value)


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
