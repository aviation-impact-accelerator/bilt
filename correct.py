"""Examples."""

from bilt.core import system
from bilt.core._system._decorator import Attribute
from bilt.core._system._dispatch import _dispatch_evaluation
from bilt.core._system._interface import Interface
from bilt.core._system._sentinel import MISSING, _Missing


@system
class Automatic:  # noqa: D101
    a: float
    b: float


class Manual(metaclass=Interface):  # noqa: D101
    __bilt_system_attributes__ = {"a": Attribute("a", float), "b": Attribute("b", float)}  # noqa: RUF012

    def __init__(self, a: float | _Missing = MISSING, b: float | _Missing = MISSING) -> None:  # noqa: D107
        self.__bilt_system_implementation__ = MISSING
        self.__bilt_system_instance_variables__ = {"a": a, "b": b}

    @property
    def a(self) -> float:  # noqa: D102
        return _dispatch_evaluation(self, self.__bilt_system_attributes__["a"])  # type: ignore[arg-type]

    @property
    def b(self) -> float:  # noqa: D102
        return _dispatch_evaluation(self, self.__bilt_system_attributes__["b"])  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:  # noqa: D105
        if not isinstance(other, Manual):
            return NotImplemented
        return {k: self.__bilt_system_instance_variables__[k] for k in self.__bilt_system_attributes__} == {
            k: other.__bilt_system_instance_variables__[k] for k in other.__bilt_system_attributes__
        }

    def __repr__(self) -> str:  # noqa: D105
        known_variables = {
            name: self.__bilt_system_instance_variables__[name]
            for name in self.__bilt_system_attributes__
            if not isinstance(self.__bilt_system_instance_variables__[name], _Missing)
        }
        formatted_variables = ", ".join(f"{name}={value}" for name, value in known_variables.items())
        return f"{self.__class__.__name__}({formatted_variables})"

    __hash__ = None  # type: ignore[assignment]
