from __future__ import annotations

import typing


class Dimension:
    __slots__ = (
        "_amount",
        "_currency",
        "_current",
        "_length",
        "_luminosity",
        "_mass",
        "_temperature",
        "_time",
    )

    _cache: typing.ClassVar[dict[tuple[int, int, int, int, int, int, int, int], typing.Self]] = {}

    def __new__(  # noqa: PLR0913
        cls,
        mass: int = 0,
        length: int = 0,
        time: int = 0,
        temperature: int = 0,
        current: int = 0,
        amount: int = 0,
        luminosity: int = 0,
        currency: int = 0,
    ) -> typing.Self:
        key = (mass, length, time, temperature, current, amount, luminosity, currency)
        if key in cls._cache:
            return cls._cache[key]
        instance = object.__new__(cls)
        cls._cache[key] = instance
        return instance

    def __init__(  # noqa: PLR0913
        self,
        mass: int = 0,
        length: int = 0,
        time: int = 0,
        temperature: int = 0,
        current: int = 0,
        amount: int = 0,
        luminosity: int = 0,
        currency: int = 0,
    ) -> None:
        self._mass = mass
        self._length = length
        self._time = time
        self._temperature = temperature
        self._current = current
        self._amount = amount
        self._luminosity = luminosity
        self._currency = currency

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
    def is_dimensionless(self) -> bool:
        return sum(abs(dimension) for dimension in self.dimensions) == 0

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

    def __mul__(self, other: typing.Self) -> typing.Self:
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
        )

    def __truediv__(self, other: typing.Self) -> typing.Self:
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
        )

    def __repr__(self) -> str:
        if self.is_dimensionless:
            return f"{self.__class__.__name__}()"
        dimensions: dict[str, int] = {}
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
        kwargs = ", ".join(f"{key}={value}" for key, value in dimensions.items())
        return f"{self.__class__.__name__}({kwargs})"

    def __str__(self) -> str:
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
