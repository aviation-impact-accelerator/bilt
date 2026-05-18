from bilt.core import system
from bilt.core._system._interface import Interface


@system
class Demo:
    a: float
    b: float
    c: float


def test_decorator() -> None:
    @system
    class Demo:
        a: float
        b: float
        c: float

    assert Demo.__name__ == "Demo"


def test_metaclass_is_interface() -> None:
    assert isinstance(Demo, Interface)


def test_has_private_registry() -> None:
    assert hasattr(Demo, "__bilt_system_registry__")
    assert Demo.__bilt_system_registry__ == set()
