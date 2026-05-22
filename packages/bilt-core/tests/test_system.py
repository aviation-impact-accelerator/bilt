import pytest

from bilt.core import (
    BiltSystemDefinitionError,
    BiltUnknownSystemAttributeError,
    Implementation,
    Interface,
    field,
    system,
    transform,
)


@system
class Demo(metaclass=Interface):
    a: float = field()
    b: float = field()
    c: float = field()


def test_decorator() -> None:
    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()
        c: float = field()

    assert Demo.__name__ == "Demo"


def test_metaclass_is_interface() -> None:
    assert isinstance(Demo, Interface)


def test_system_requires_interface_metaclass() -> None:
    with pytest.raises(
        BiltSystemDefinitionError,
        match="Cannot apply `@system` to 'NotAnInterface' because it does not declare `metaclass=Interface`",
    ):

        @system  # ty: ignore[invalid-argument-type]
        class NotAnInterface:
            pass


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


@pytest.mark.parametrize(
    ("args"),
    ({"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 1.0}),
)
def test_eq_same_inputs(args: dict[str, float]) -> None:
    assert Demo(**args) == Demo(**args)


@pytest.mark.parametrize(
    ("left", "right"),
    (
        ({"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 1.0, "b": 2.0, "c": 4.0}),
        ({"a": 1.0}, {"a": 2.0}),
        ({"a": 1.0}, {"b": 2.0}),
    ),
)
def test_eq_different_inputs(left: dict[str, float], right: dict[str, float]) -> None:
    assert Demo(**left) != Demo(**right)


def test_eq_different_type() -> None:
    assert Demo(a=1.0) != object()


@pytest.mark.parametrize(
    ("instance", "expected_repr"),
    (
        (Demo(a=1.0), "Demo(a=1.0)"),
        (Demo(a=1.0, c=3.0), "Demo(a=1.0, c=3.0)"),
        (Demo(a=1.0, b=2.0, c=3.0), "Demo(a=1.0, b=2.0, c=3.0)"),
    ),
)
def test_repr(instance: Demo, expected_repr: str) -> None:
    assert repr(instance) == expected_repr


def test_not_hashable() -> None:
    with pytest.raises(TypeError, match="unhashable type"):
        hash(Demo(a=1.0))


def test_register_implementation() -> None:
    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()
        c: float = field()

    @transform
    def c(a: float, b: float) -> float:
        return a + b

    assert Demo.__bilt_system_registry__ == set()

    basic = Implementation(Demo, [c])
    assert basic in Demo.__bilt_system_registry__
