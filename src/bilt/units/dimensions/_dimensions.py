import typing

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.units._metadata import Metadata


@typing.final
class Dimension:
    __slots__ = (
        "_amount",
        "_currency",
        "_current",
        "_length",
        "_luminosity",
        "_mass",
        "_tagged",
        "_temperature",
        "_time",
    )

    _cache: typing.ClassVar[
        dict[
            tuple[int, int, int, int, int, int, int, int, frozenset[tuple[tuple[Dimension, Metadata], int]]],
            typing.Self,
        ]
    ] = {}

    def __new__(  # noqa: PLR0913, PLR0917
        cls,
        mass: int = 0,
        length: int = 0,
        time: int = 0,
        temperature: int = 0,
        current: int = 0,
        amount: int = 0,
        luminosity: int = 0,
        currency: int = 0,
        *,
        tagged: collections.abc.Mapping[tuple[Dimension, Metadata], int] | None = None,
    ) -> typing.Self:
        key = (
            mass,
            length,
            time,
            temperature,
            current,
            amount,
            luminosity,
            currency,
            frozenset((factor, order) for factor, order in tagged.items() if order != 0) if tagged else frozenset(),
        )
        if key in cls._cache:
            return cls._cache[key]
        instance = object.__new__(cls)
        cls._cache[key] = instance
        return instance

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        mass: int = 0,
        length: int = 0,
        time: int = 0,
        temperature: int = 0,
        current: int = 0,
        amount: int = 0,
        luminosity: int = 0,
        currency: int = 0,
        *,
        tagged: collections.abc.Mapping[tuple[Dimension, Metadata], int] | None = None,
    ) -> None:
        self._mass = mass
        self._length = length
        self._time = time
        self._temperature = temperature
        self._current = current
        self._amount = amount
        self._luminosity = luminosity
        self._currency = currency
        self._tagged = {factor: order for factor, order in tagged.items() if order != 0} if tagged else {}

    @property
    def mass(self) -> int:
        return self._mass

    @property
    def length(self) -> int:
        return self._length

    @property
    def time(self) -> int:
        return self._time

    @property
    def temperature(self) -> int:
        return self._temperature

    @property
    def current(self) -> int:
        return self._current

    @property
    def amount(self) -> int:
        return self._amount

    @property
    def luminosity(self) -> int:
        return self._luminosity

    @property
    def currency(self) -> int:
        return self._currency

    @property
    def tagged(self) -> collections.abc.Mapping[tuple[Dimension, Metadata], int]:
        return self._tagged

    @property
    def is_dimensionless(self) -> bool:
        return sum(abs(dimension) for dimension in self.dimensions) == 0 and not self.tagged

    @property
    def is_primary(self) -> bool:
        return sum(abs(dimension) for dimension in self.dimensions) == 1

    @property
    def is_derived(self) -> bool:
        return sum(abs(dimension) for dimension in self.dimensions) >= 2  # noqa: PLR2004

    @property
    def dimensions(self) -> tuple[int, int, int, int, int, int, int, int]:
        return (
            self.mass,
            self.length,
            self.time,
            self.temperature,
            self.current,
            self.amount,
            self.luminosity,
            self.currency,
        )

    def __mul__(self, other: Dimension) -> typing.Self:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__class__(
            mass=self.mass + other.mass,
            length=self.length + other.length,
            time=self.time + other.time,
            temperature=self.temperature + other.temperature,
            current=self.current + other.current,
            amount=self.amount + other.amount,
            luminosity=self.luminosity + other.luminosity,
            currency=self.currency + other.currency,
            tagged={
                factor: self.tagged.get(factor, 0) + other.tagged.get(factor, 0)
                for factor in (*self.tagged, *other.tagged)
            }
            if self.tagged or other.tagged
            else None,
        )

    def __truediv__(self, other: Dimension) -> typing.Self:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__class__(
            mass=self.mass - other.mass,
            length=self.length - other.length,
            time=self.time - other.time,
            temperature=self.temperature - other.temperature,
            current=self.current - other.current,
            amount=self.amount - other.amount,
            luminosity=self.luminosity - other.luminosity,
            currency=self.currency - other.currency,
            tagged={
                factor: self.tagged.get(factor, 0) - other.tagged.get(factor, 0)
                for factor in (*self.tagged, *other.tagged)
            }
            if self.tagged or other.tagged
            else None,
        )

    def __pow__(self, other: int) -> typing.Self:
        if not isinstance(other, int):
            return NotImplemented
        return self.__class__(
            mass=self.mass * other,
            length=self.length * other,
            time=self.time * other,
            temperature=self.temperature * other,
            current=self.current * other,
            amount=self.amount * other,
            luminosity=self.luminosity * other,
            currency=self.currency * other,
            tagged={factor: order * other for factor, order in self.tagged.items()} if self.tagged else None,
        )

    def __repr__(self) -> str:
        dimensions: dict[str, object] = {}
        if self.mass != 0:
            dimensions["mass"] = self.mass
        if self.length != 0:
            dimensions["length"] = self.length
        if self.time != 0:
            dimensions["time"] = self.time
        if self.temperature != 0:
            dimensions["temperature"] = self.temperature
        if self.current != 0:
            dimensions["current"] = self.current
        if self.amount != 0:
            dimensions["amount"] = self.amount
        if self.luminosity != 0:
            dimensions["luminosity"] = self.luminosity
        if self.currency != 0:
            dimensions["currency"] = self.currency
        if self.tagged:
            dimensions["tagged"] = dict(self.tagged)
        kwargs = ", ".join(f"{key}={value}" for key, value in dimensions.items())
        return f"{self.__class__.__name__}({kwargs})"

    def __str__(self) -> str:
        name = _NAMES.get(self)
        if name is None:
            return self._exponent_form()
        return f"{name} ({self._exponent_form()})"

    def _exponent_form(self) -> str:
        def format_term(dimension: str, index: int) -> str:
            match index:
                case 0:
                    return ""
                case 1:
                    return dimension
                case _ if index < 0:
                    return f"{dimension}^({index})"
                case _:
                    return f"{dimension}^{index}"

        if self.is_dimensionless:
            return "[Dimensionless]"
        dimensions = {
            "[Mass]": self.mass,
            "[Length]": self.length,
            "[Time]": self.time,
            "[Temperature]": self.temperature,
            "[Current]": self.current,
            "[Amount]": self.amount,
            "[Luminosity]": self.luminosity,
            "[Currency]": self.currency,
        } | {
            f"({form})|{tag}" if " " in form else f"{form}|{tag}": order
            for form, tag, order in sorted(
                (dimension._exponent_form(), str(tag), order)  # noqa: SLF001
                for (dimension, tag), order in self.tagged.items()
            )
        }
        return " ".join(format_term(dimension, index) for dimension, index in dimensions.items() if index != 0)


MASS = Dimension(mass=1)
LENGTH = Dimension(length=1)
TIME = Dimension(time=1)
TEMPERATURE = Dimension(temperature=1)
CURRENT = Dimension(current=1)
AMOUNT = Dimension(amount=1)
LUMINOSITY = Dimension(luminosity=1)
CURRENCY = Dimension(currency=1)

SI_DIMENSIONS = (MASS, LENGTH, TIME, TEMPERATURE, CURRENCY, AMOUNT)
NON_SI_DIMENSION = (CURRENCY,)
DIMENSIONS = SI_DIMENSIONS + NON_SI_DIMENSION

DIMENSIONLESS = Dimension()
AREA = LENGTH**2
VOLUME = LENGTH**3
FORCE = MASS * LENGTH / TIME**2
ENERGY = MASS * LENGTH**2 / TIME**2
POWER = MASS * LENGTH**2 / TIME**3
PRESSURE = MASS / (LENGTH * TIME**2)
DYNAMIC_VISCOSITY = MASS / (LENGTH * TIME)
CHARGE = TIME * CURRENT
VOLTAGE = MASS * LENGTH**2 / (TIME**3 * CURRENT)
RESISTANCE = MASS * LENGTH**2 / (TIME**3 * CURRENT**2)
CONDUCTANCE = TIME**3 * CURRENT**2 / (MASS * LENGTH**2)

_NAMES: dict[Dimension, str] = {
    MASS: "mass",
    LENGTH: "length",
    TIME: "time",
    TEMPERATURE: "temperature",
    CURRENT: "current",
    AMOUNT: "amount",
    LUMINOSITY: "luminosity",
    CURRENCY: "currency",
    DIMENSIONLESS: "dimensionless",
    AREA: "area",
    VOLUME: "volume",
    FORCE: "force",
    ENERGY: "energy",
    POWER: "power",
    PRESSURE: "pressure",
    DYNAMIC_VISCOSITY: "dynamic viscosity",
    CHARGE: "charge",
    VOLTAGE: "voltage",
    RESISTANCE: "resistance",
    CONDUCTANCE: "conductance",
}


class _Absolute:
    def __repr__(self) -> str:
        return "ABSOLUTE"


ABSOLUTE = _Absolute()
