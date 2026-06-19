import dataclasses
import functools
import operator
import typing

from ._category import UnitCategory
from ._dimensions import (
    AMOUNT,
    CURRENCY,
    CURRENT,
    LENGTH,
    LUMINOSITY,
    MASS,
    TEMPERATURE,
    TIME,
    Dimension,
)
from ._errors import UnitError
from ._prefix import PREFIX_NAMES, Prefix
from ._quantities import Quantity

if typing.TYPE_CHECKING:
    from ._metadata import Metadata


class UnitOrder:
    __slots__ = ("_dimension", "_order", "_unit")

    _cache: typing.ClassVar[dict[tuple[Unit, int], typing.Self]] = {}

    def __new__(cls, unit: Unit, order: int) -> typing.Self:
        key = (unit, order)
        if key in cls._cache:
            return cls._cache[key]

        instance = object.__new__(cls)
        cls._cache[key] = instance
        return instance

    def __init__(self, unit: Unit, order: int) -> None:
        self._unit = unit
        self._order = order
        self._dimension = self.unit.dimension**self.order

    @property
    def unit(self) -> Unit:
        return self._unit

    @property
    def order(self) -> int:
        return self._order

    @property
    def dimension(self) -> Dimension:
        return self._dimension

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.unit is other.unit) and (self.order == other.order)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.unit is other.unit:
            return self.order < other.order
        return self.unit < other.unit

    def __hash__(self) -> int:
        return id(self)

    def __str__(self) -> str:
        if self.order == 1:
            return self.unit.name
        if self.order >= 2:  # noqa: PLR2004
            return f"{self.unit.name}^{self.order}"
        return f"{self.unit.name}^({self.order})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(unit={self.unit}, order={self.order})"


class UnitComposition:
    __slots__ = ("_composition", "_dimension", "_scale")

    _cache: typing.ClassVar[dict[tuple[tuple[UnitOrder, ...], float], typing.Self]] = {}
    _composition: tuple[UnitOrder, ...]
    _scale: float
    _dimension: Dimension

    def __new__(cls, *composition: UnitOrder, scale: float = 1.0) -> typing.Self:
        unit_to_order_mapping = {term.unit: 0 for term in composition if term.unit is not DIMENSIONLESS}
        for term in composition:
            if term.unit is not DIMENSIONLESS:
                unit_to_order_mapping[term.unit] += term.order
        composition = tuple(
            sorted(UnitOrder(unit, order) for unit, order in unit_to_order_mapping.items() if order != 0)
        )
        key = (composition, scale)
        if key in cls._cache:
            return cls._cache[key]

        instance = object.__new__(cls)
        instance._composition = composition  # noqa: SLF001
        instance._scale = scale  # noqa: SLF001
        instance._dimension = functools.reduce(  # noqa: SLF001
            operator.mul,
            (term.dimension for term in instance._composition),  # noqa: SLF001
            Dimension(),
        )
        cls._cache[key] = instance
        return instance

    @property
    def composition(self) -> tuple[UnitOrder, ...]:
        return self._composition

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def dimension(self) -> Dimension:
        return self._dimension

    @property
    def si_base_units(self) -> typing.Self:
        if all(term.unit.is_si_base_unit for term in self.composition):
            return self

        terms = []
        scale = self.scale
        for term in self.composition:
            if term.unit.is_si_base_unit:
                terms.append(term)
            else:
                si_base_units = (term.unit.relation**term.order).si_base_units
                terms.extend(list(si_base_units.composition))
                scale *= si_base_units.scale

        return self.__class__(*terms, scale=scale)

    @property
    def is_coherent(self) -> bool:
        if len(self) == 0:
            return True
        return all(term.unit.is_coherent for term in self.composition) and self.scale == 1.0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self._composition == other._composition:
            return self._scale == other._scale
        self_si_base_units = self.si_base_units
        other_si_base_units = other.si_base_units
        return (self_si_base_units._composition == other_si_base_units._composition) and (
            self_si_base_units._scale == other_si_base_units._scale
        )

    def __mul__(self, other: typing.Self) -> typing.Self:
        if not isinstance(other, self.__class__):
            return NotImplemented

        composition = {term.unit: term.order for term in self.composition}
        for term in other.composition:
            if term.unit in composition:
                composition[term.unit] += term.order
            else:
                composition[term.unit] = term.order

        return self.__class__(
            *[UnitOrder(unit, order) for unit, order in composition.items() if order != 0],
            scale=self.scale * other.scale,
        )

    def __rmul__(self, other: float) -> typing.Self:
        if not isinstance(other, float):
            return NotImplemented
        if other == 0.0:
            message = f"Cannot multiply unit composition {self} by {other} as result is ambiguous"
            raise UnitError(message)
        return self.__class__(*self.composition, scale=self.scale * other)

    def __truediv__(self, other: typing.Self) -> typing.Self:
        if not isinstance(other, self.__class__):
            return NotImplemented

        composition = {term.unit: term.order for term in self.composition}
        for term in other.composition:
            if term.unit in composition:
                composition[term.unit] -= term.order
            else:
                composition[term.unit] = -term.order

        return self.__class__(
            *[UnitOrder(unit, order) for unit, order in composition.items() if order != 0],
            scale=self.scale / other.scale,
        )

    def __rtruediv__(self, other: typing.Literal[1]) -> typing.Self:
        if not isinstance(other, int) and other == 1:
            return NotImplemented
        return self.__class__(
            *[UnitOrder(term.unit, -term.order) for term in self.composition],
            scale=(1.0 / self.scale),
        )

    def __pow__(self, other: int) -> typing.Self:
        if other == 0:
            return self.__class__()
        return self.__class__(
            *[UnitOrder(term.unit, term.order * other) for term in self._composition],
            scale=self.scale**other,
        )

    def __hash__(self) -> int:
        return id(self)

    def __len__(self) -> int:
        return len(self._composition)

    def __str__(self) -> str:
        def format_term(unit: str, order: int) -> str:
            match order:
                case 0:
                    return ""
                case 1:
                    return str(unit)
                case _ if order < 0:
                    return f"{unit!s}^({order})"
                case _:
                    return f"{unit!s}^{order}"

        composition = " ".join(format_term(term.unit.name, term.order) for term in self.composition)
        if not composition:
            composition = "DIMENSIONLESS"
        if self.scale == 1.0:
            return composition
        return f"{self.scale} {composition}"

    def __repr__(self) -> str:
        if not self.composition:
            return f"{self.__class__.__name__}(scale={self.scale!r})"
        return f"{self.__class__.__name__}({', '.join(repr(term) for term in self.composition)}, scale={self.scale})"


class Unit:
    __slots__ = ("_category", "_dimension", "_metadata", "_name", "_offset", "_prefix", "_relation")

    _cache: typing.ClassVar[dict[tuple[str, UnitComposition, Metadata | None], typing.Self]] = {}
    _name_cache: typing.ClassVar[dict[str, typing.Self]] = {}
    _absolute_cache: typing.ClassVar[dict[Unit, Unit]] = {}

    _name: str
    _relation: UnitComposition
    _prefix: Prefix | None
    _metadata: Metadata | None
    _scale: Unit
    _offset: UnitComposition | None
    _dimension: Dimension
    _category: UnitCategory

    def __new__(
        cls,
        relation: UnitComposition,
        metadata: Metadata | None = None,
    ) -> typing.Self:
        name = _create_name(relation, metadata)
        key = (name, relation, metadata)
        if key in cls._cache:
            return cls._cache[key]

        if (
            len(relation) == 1
            and relation.composition[0].order == 1
            and metadata is None
            and (relation.composition[0].unit.relation.scale == relation.scale or relation.scale == 1.0)
        ):
            return typing.cast("typing.Self", relation.composition[0].unit)

        instance = object.__new__(cls)
        instance._name = name  # noqa: SLF001
        instance._prefix = None  # noqa: SLF001
        instance._relation = relation  # noqa: SLF001
        instance._offset = None  # noqa: SLF001
        instance._metadata = metadata  # noqa: SLF001
        instance._dimension = instance._relation.dimension  # noqa: SLF001
        instance._category = UnitCategory.DERIVED  # noqa: SLF001

        cls._cache[key] = instance

        return instance

    @classmethod
    def new_named(
        cls, name: str, relation: Unit | UnitComposition | Quantity, prefix: Prefix | None = None
    ) -> typing.Self:
        if prefix is not None and not name.startswith(PREFIX_NAMES[prefix]):
            message = f"Unit with name {name!r} does not start with its defined prefix of {prefix}"
            raise UnitError(message)

        match relation:
            case Unit():
                relation = relation.relation
            case UnitComposition():
                pass
            case Quantity():
                quantity = relation
                if quantity.unit.category in (UnitCategory.BASE, UnitCategory.NAMED):
                    relation = UnitComposition(
                        UnitOrder(quantity.unit, 1),
                        scale=quantity.value,
                    )
                else:
                    relation = UnitComposition(
                        *quantity.unit.relation.composition,
                        scale=quantity.value * quantity.unit.relation.scale,
                    )
            case _:
                typing.assert_never(relation)

        if relation is UnitComposition(UnitOrder(kilogram, 1)):
            prefix_scale = 0.001 if prefix is None else 10 ** (prefix.value - 3)
        else:
            prefix_scale = (1.0 if prefix is None else 10**prefix.value) / functools.reduce(
                operator.mul,
                (
                    1.0 if unit_order.unit.prefix is None else unit_order.unit.relation.scale
                    for unit_order in relation.composition
                ),
                1.0,
            )
        relation = UnitComposition(*relation.composition, scale=relation.scale * prefix_scale)

        key = (name, relation, None)
        if key in cls._cache:
            return cls._cache[key]

        if name in cls._name_cache:
            instance = cls._name_cache[name]
            message = (
                f"Cannot instantiate unit {name!r} because relation {relation!r} does not "
                f"match the relation {instance.relation!r} of the cached version of the unit "
                f"with the same name"
            )
            raise UnitError(message)

        instance = object.__new__(cls)
        instance._name = name  # noqa: SLF001
        instance._relation = relation  # noqa: SLF001
        instance._offset = None  # noqa: SLF001
        instance._prefix = prefix  # noqa: SLF001
        instance._metadata = None  # noqa: SLF001
        instance._dimension = instance._relation.dimension  # noqa: SLF001
        instance._category = UnitCategory.NAMED  # noqa: SLF001

        cls._cache[(name, relation, None)] = instance
        cls._name_cache[name] = instance

        return instance

    @classmethod
    def new_absolute(
        cls, name: str, scale: Unit, zero: Quantity | None = None, metadata: Metadata | None = None
    ) -> typing.Self:
        relation = UnitComposition(UnitOrder(scale, 1))
        offset: UnitComposition | None
        match zero:
            case Quantity():
                offset = UnitComposition(UnitOrder(zero.unit, 1), scale=zero.value)
            case None:
                offset = None
            case _:
                typing.assert_never(zero)
        key = (name, relation, metadata)
        if key in cls._cache:
            instance = cls._cache[key]
            if offset != instance.offset:
                message = (
                    f"Cannot instantiate new `{cls}` with zero {zero} because one already exists "
                    f"with the same name {name} and scale {scale} but a different zero of "
                    f"{instance.offset}"
                )
                raise UnitError(message)

            return instance

        if scale in cls._absolute_cache:
            message = (
                f"Cannot instantiate new `{cls}` with scale {scale} because one already exists with the same scale"
            )
            raise UnitError(message)

        instance = object.__new__(cls)
        instance._name = name  # noqa: SLF001
        instance._relation = relation  # noqa: SLF001
        instance._offset = offset  # noqa: SLF001
        instance._prefix = None  # noqa: SLF001
        instance._metadata = metadata  # noqa: SLF001
        instance._dimension = scale.dimension  # noqa: SLF001
        instance._category = UnitCategory.ABSOLUTE  # noqa: SLF001

        cls._cache[key] = instance
        cls._name_cache[name] = instance
        cls._absolute_cache[scale] = instance

        return instance

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> Dimension:
        return self._dimension

    @property
    def prefix(self) -> Prefix | None:
        return self._prefix

    @property
    def relation(self) -> UnitComposition:
        return self._relation

    @property
    def offset(self) -> UnitComposition | None:
        return self._offset

    @property
    def metadata(self) -> Metadata | None:
        return self._metadata

    @property
    def category(self) -> UnitCategory:
        return self._category

    @property
    def is_dimensionless(self) -> bool:
        return self.dimension is Dimension()

    @property
    def is_si_base_unit(self) -> bool:
        return self in BASE_UNIT_SYSTEM

    @property
    def is_coherent(self) -> bool:
        if self.is_si_base_unit:
            return True
        return self.relation.is_coherent

    @property
    def is_si_derived_unit(self) -> bool:
        return not self.is_si_base_unit and self.is_coherent

    def strip_metadata(self) -> typing.Self:
        if self.metadata is not None:
            return self.__class__(self.relation)
        return self.__class__(
            UnitComposition(
                *[
                    UnitOrder(self.__class__(unit_order.unit.relation), unit_order.order)
                    for unit_order in self.relation.composition
                ],
                scale=self.relation.scale,
            )
        )

    def __abs__(self) -> Unit:
        if self not in self.__class__._absolute_cache:  # noqa: SLF001
            message = f"No absolute unit corresponding to {self} exists"
            raise UnitError(message)
        return self.__class__._absolute_cache[self]  # noqa: SLF001

    @typing.overload
    def __mul__(self, other: typing.Self) -> typing.Self: ...

    @typing.overload
    def __mul__(self, other: Quantity) -> Quantity: ...

    def __mul__(self, other: typing.Self | Quantity) -> typing.Self | Quantity:
        if isinstance(other, self.__class__):
            left = (
                self.relation
                if self.category is UnitCategory.DERIVED and self.metadata is None
                else UnitComposition(UnitOrder(self, 1))
            )
            right = (
                other.relation
                if other.category is UnitCategory.DERIVED and other.metadata is None
                else UnitComposition(UnitOrder(other, 1))
            )
            return self.__class__(left * right)
        if isinstance(other, Quantity):
            # pyrefly: ignore[unsupported-operation]
            return Quantity(other.value, self * other.unit)
        return NotImplemented

    def __rmul__(self, other: Quantity | float) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity(other.value, other.unit * self)
        if isinstance(other, float):
            return Quantity(other, self)
        return NotImplemented

    @typing.overload
    def __truediv__(self, other: typing.Self) -> typing.Self: ...

    @typing.overload
    def __truediv__(self, other: Quantity) -> Quantity: ...

    def __truediv__(self, other: typing.Self | Quantity) -> typing.Self | Quantity:
        if isinstance(other, self.__class__):
            numerator = (
                self.relation
                if self.category is UnitCategory.DERIVED and self.metadata is None
                else UnitComposition(UnitOrder(self, 1))
            )
            denominator = (
                other.relation
                if other.category is UnitCategory.DERIVED and other.metadata is None
                else UnitComposition(UnitOrder(other, 1))
            )
            return self.__class__(numerator / denominator)
        if isinstance(other, Quantity):
            # pyrefly: ignore[unsupported-operation]
            return Quantity(1.0 / other.value, self / other.unit)
        return NotImplemented

    @typing.overload
    def __rtruediv__(self, other: typing.Literal[1]) -> typing.Self: ...

    @typing.overload
    def __rtruediv__(self, other: Quantity) -> Quantity: ...

    def __rtruediv__(self, other: Quantity | typing.Literal[1]) -> typing.Self | Quantity:
        if isinstance(other, Quantity):
            return Quantity(other.value, other.unit / self)
        if isinstance(other, int) and other == 1:
            return self**-1
        return NotImplemented

    def __pow__(self, other: float) -> typing.Self:
        if isinstance(other, int):
            base = (
                UnitComposition(UnitOrder(self, 1))
                if (len(self.relation) == 1 and self.relation.composition[0].order == 1)
                else self.relation
            )
            return self.__class__(base**other)
        if isinstance(other, float) and self is DIMENSIONLESS:
            return self
        return NotImplemented

    def to_si_base_units(self) -> typing.Self:
        if self.is_si_base_unit:
            return self
        return self.__class__(self.relation.si_base_units, self.metadata)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self.name == other.name:
            if self.metadata is None:
                return True
            if other.metadata is None:
                return False
            return self.metadata < other.metadata
        return self.name < other.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self.name == other.name:
            return True
        if self.dimension != other.dimension:
            return False
        if self.relation.composition == other.relation.composition:
            return (
                (self.relation.scale == other.relation.scale)
                and (self.prefix == other.prefix)
                and (self.metadata == other.metadata)
            )
        return self.to_si_base_units() == other.to_si_base_units()

    def __hash__(self) -> int:
        return id(self)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self._name!r}, relation={self.relation!r}, "
            f"prefix={self.prefix!r}, metadata={self.metadata!r})"
        )


def _create_name(relation: UnitComposition, metadata: Metadata | None) -> str:
    def format_term(term: UnitOrder) -> str:
        if abs(term.order) == 1:
            return str(term.unit)
        return f"{term.unit}^{abs(term.order)}"

    if metadata is not None and len(relation.composition) == 1 and relation.composition[0].order == 1:
        return (
            f"{relation.composition[0].unit.name}|{metadata}"
            if relation.scale == 1.0
            else f"<{relation.scale}> {relation.composition[0].unit.name}|{metadata}"
        )

    numerator = " ".join(format_term(term) for term in relation.composition if term.order >= 1)
    denominator = " ".join(format_term(term) for term in relation.composition if term.order <= -1)
    match (numerator, denominator):
        case "", "":
            name = "DIMENSIONLESS"
        case n, "" if n != "":
            name = numerator
        case "", d if d != "":
            name = f"1 / {denominator}"
        case _:
            name = f"{numerator} / {denominator}"

    if metadata is not None:
        name = f"({name})|{metadata!s}"
    if relation.scale != 1.0:
        name = f"<{relation.scale}> {name}"
    return name


def _define_base_unit(name: str, dimension: Dimension, prefix: Prefix | None = None) -> Unit:
    base_unit = object.__new__(Unit)
    base_unit._name = name  # noqa: SLF001
    base_unit._prefix = prefix  # noqa: SLF001
    base_unit._dimension = dimension  # noqa: SLF001
    relation = UnitComposition(UnitOrder(base_unit, 1))
    base_unit._relation = relation  # noqa: SLF001
    base_unit._metadata = None  # noqa: SLF001
    base_unit._offset = None  # noqa: SLF001
    base_unit._category = UnitCategory.BASE  # noqa: SLF001

    Unit._cache[(name, relation, None)] = base_unit  # noqa: SLF001
    Unit._name_cache[name] = base_unit  # noqa: SLF001

    return base_unit


DIMENSIONLESS = Unit(UnitComposition())

kilogram = _define_base_unit("kilogram", MASS, prefix=Prefix.KILO)
meter = _define_base_unit("meter", LENGTH)
second = _define_base_unit("second", TIME)
kelvin = _define_base_unit("kelvin", TEMPERATURE)
ampere = _define_base_unit("ampere", CURRENT)
mole = _define_base_unit("mole", AMOUNT)
candela = _define_base_unit("candela", LUMINOSITY)
_USD2025 = _define_base_unit("USD2025", CURRENCY)


@dataclasses.dataclass(frozen=True, slots=True)
class UnitSystem:
    mass: Unit
    length: Unit
    time: Unit
    temperature: Unit
    current: Unit
    amount: Unit
    luminosity: Unit
    currency: Unit

    @property
    def dimension_to_base_unit_mapping(self) -> dict[Dimension, Unit]:
        return {
            MASS: self.mass,
            LENGTH: self.length,
            TIME: self.time,
            TEMPERATURE: self.temperature,
            CURRENT: self.current,
            AMOUNT: self.amount,
            LUMINOSITY: self.luminosity,
            CURRENCY: self.currency,
        }

    def __contains__(self, item: Unit) -> bool:
        for si_base_unit in self.dimension_to_base_unit_mapping.values():  # noqa: SIM110
            if item is si_base_unit:
                return True
        return False

    def __getitem__(self, dimension: Dimension) -> Unit:
        return self.dimension_to_base_unit_mapping[dimension]


BASE_UNIT_SYSTEM = UnitSystem(
    mass=kilogram,
    length=meter,
    time=second,
    temperature=kelvin,
    current=ampere,
    amount=mole,
    luminosity=candela,
    currency=_USD2025,
)
