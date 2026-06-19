import typing

from ._errors import MetadataError
from ._quantities import Quantity
from ._units import Unit, UnitComposition, UnitOrder


class Metadata:
    __slots__ = ("_value",)

    _cache: typing.ClassVar[dict[str, typing.Self]] = {}

    def __new__(cls, value: str) -> typing.Self:
        if value in cls._cache:
            return cls._cache[value]

        if value == "":
            message = "Metadata must be a non-empty value"
            raise MetadataError(message)

        instance = object.__new__(cls)
        cls._cache[value] = instance
        return instance

    def __init__(self, value: str) -> None:
        self._value = value

    @property
    def value(self) -> str:
        return self._value

    @typing.overload
    def __rmul__(self, other: Unit) -> Unit: ...

    @typing.overload
    def __rmul__(self, other: Quantity) -> Quantity: ...

    def __rmul__(self, other: Unit | Quantity) -> Unit | Quantity:
        if isinstance(other, Unit):
            return Unit(UnitComposition(UnitOrder(other, 1)), metadata=self)
        if isinstance(other, Quantity):
            return Quantity(other.value, other.unit * self)
        return NotImplemented

    def __lt__(self, other: Metadata | None) -> bool:
        if other is None:
            return False
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.value < other.value

    def __gt__(self, other: Metadata | None) -> bool:
        if other is None:
            return True
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.value > other.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(value={self.value!r})"

    def __str__(self) -> str:
        return str(self.value)
