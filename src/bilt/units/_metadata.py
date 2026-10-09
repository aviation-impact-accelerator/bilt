import typing

from bilt.units._errors import BiltMetadataError

from ._quantities import Quantity
from ._units import Unit, UnitComposition, UnitOrder
from .dimensions._dimensions import Dimension


@typing.final
class Metadata:
    __slots__ = ("_value",)

    _cache: typing.ClassVar[dict[str, typing.Self]] = {}

    def __new__(cls, value: str) -> typing.Self:
        if value in cls._cache:
            return cls._cache[value]

        if value == "":
            message = "Metadata must be a non-empty value"
            raise BiltMetadataError(message)

        instance = object.__new__(cls)
        cls._cache[value] = instance
        return instance

    def __init__(self, value: str) -> None:
        self._value = value

    @property
    def value(self) -> str:
        return self._value

    @typing.overload
    def __rmul__(self, other: Dimension) -> Dimension: ...

    @typing.overload
    def __rmul__(self, other: Unit) -> Unit: ...

    @typing.overload
    def __rmul__(self, other: Quantity) -> Quantity: ...

    def __rmul__(self, other: Dimension | Unit | Quantity) -> Dimension | Unit | Quantity:
        if isinstance(other, Dimension):
            return Dimension(tagged={(other, self): 1})
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
