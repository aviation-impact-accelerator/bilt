import pytest

from bilt.core import BiltUnknownSystemAttributeError, field, system
from bilt.core._system._interface import Interface


@system
class Demo:
    a: float = field()
    b: float = field()
    c: float = field()


def test_decorator() -> None:
    @system
    class Demo:
        a: float = field()
        b: float = field()
        c: float = field()

    assert Demo.__name__ == "Demo"


def test_metaclass_is_interface() -> None:
    assert isinstance(Demo, Interface)


def test_has_private_registry() -> None:
    assert hasattr(Demo, "__bilt_system_registry__")
    assert Demo.__bilt_system_registry__ == set()


def test_has_all_fields_as_attributes() -> None:
    assert hasattr(Demo, "a")
    assert hasattr(Demo, "b")
    assert hasattr(Demo, "c")


def test_can_instantiate_with_all_fields() -> None:
    demo = Demo(a=1.0, b=2.0, c=3.0)
    assert isinstance(demo, Demo)


def test_can_instantiate_with_a_subset_of_fields() -> None:
    demo = Demo(a=1.0)
    assert isinstance(demo, Demo)


@pytest.mark.xfail(reason="Attribute access/evaluation logic not implemented yet.")
def test_can_access_set_field() -> None:
    a = 1.0
    b = 2.0
    c = 3.0
    demo = Demo(a=a, b=b, c=c)
    assert demo.a == a
    assert demo.b == b
    assert demo.c == c


@pytest.mark.xfail(reason="Attribute access/evaluation logic not implemented yet.")
def test_cannot_access_unset_field() -> None:
    a = 1.0
    demo = Demo(a=a)
    assert demo.a == a
    with pytest.raises(
        BiltUnknownSystemAttributeError,
        match="Cannot evaluate attribute 'b' because it was not set during initialization",
    ):
        _ = demo.b
    with pytest.raises(
        BiltUnknownSystemAttributeError,
        match="Cannot evaluate attribute 'c' because it was not set during initialization",
    ):
        _ = demo.c
