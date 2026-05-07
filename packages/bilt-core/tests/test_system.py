import pytest

from bilt.core._system._decorator import system
from bilt.core._system._error import BiltSystemImplementationSelectionError, BiltUnknownSystemAttributeError
from bilt.core._system._interface import Implementation, Interface
from bilt.core._transform import transform


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


def test_register_implementation() -> None:
    @system
    class Demo:
        a: float
        b: float
        c: float

    @transform
    def c(a: float, b: float) -> float:
        return a + b

    assert Demo.__bilt_system_registry__ == set()

    basic = Implementation(Demo, [c])
    assert basic in Demo.__bilt_system_registry__


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


def test_can_access_set_field() -> None:
    a = 1.0
    b = 2.0
    c = 3.0
    demo = Demo(a=a, b=b, c=c)
    assert demo.a == a
    assert demo.b == b
    assert demo.c == c


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


def test_cannot_set_implementation_from_different_system_on_instance() -> None:
    @system
    class Demo:
        a: float
        b: float
        c: float

    @system
    class Other:
        d: float
        e: float

    @transform
    def c(a: float, b: float) -> float:
        return a + b

    other = Implementation(Other, [])

    a = 1.0
    b = 2.0
    demo = Demo(a=a, b=b)

    with pytest.raises(
        BiltSystemImplementationSelectionError,
        match="Cannot use the provided implementation with the 'Demo' instance as this is from the 'Other' system",
    ):
        demo.with_implementation(other)


def test_cannot_reset_implementation_on_system_instance() -> None:
    @system
    class Demo:
        a: float
        b: float
        c: float

    @transform
    def c(a: float, b: float) -> float:
        return a + b

    basic = Implementation(Demo, [c])

    a = 1.0
    b = 2.0
    demo = Demo(a=a, b=b)

    demo.with_implementation(basic)
    with pytest.raises(
        BiltSystemImplementationSelectionError,
        match="Cannot use the provided implementation with the 'Demo' instance as it is already using an implementation",
    ):
        demo.with_implementation(basic)


def test_evaluation_with_implementation_set_on_system_instance() -> None:
    @system
    class Demo:
        a: float
        b: float
        c: float

    @transform
    def c(a: float, b: float) -> float:
        return a + b

    basic = Implementation(Demo, [c])

    a = 1.0
    b = 2.0
    demo = Demo(a=a, b=b)

    demo.with_implementation(basic)

    expected_c = 3.0
    assert demo.c == expected_c


def test_evaluation_with_default_implementation_on_system() -> None:
    @system
    class Demo:
        a: float
        b: float
        c: float

    @transform
    def c(a: float, b: float) -> float:
        return a + b

    basic = Implementation(Demo, [c])

    a = 1.0
    b = 2.0
    demo = Demo(a=a, b=b)

    expected_c = 3.0
    with basic:
        assert demo.c == expected_c
