import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from bilt import (
    Implementation,
    Interface,
    Tensor,
    dim,
    field,
    system,
    tensor,
    transform,
)
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.errors import (
    BiltAnalysisError,
    BiltSystemDefinitionError,
    BiltUnknownSystemAttributeError,
)

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


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


def test_evaluation_with_single_implementation() -> None:
    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()
        c: float = field()

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


@pytest.mark.parametrize(
    ("backend"),
    (
        PYTHON_BACKEND,
        JAX_BACKEND,
    ),
)
def test_evaluation_with_single_implementation_and_backend(backend: Backend) -> None:
    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()
        c: float = field()
        d: float = field()
        e: float = field()

    @transform
    def c(a: float, b: float) -> float:
        return a + b

    @transform
    def d(a: float, c: float) -> float:
        return a + c

    @transform
    def e(c: float, d: float) -> float:
        return c + d

    basic = Implementation(Demo, [c, d, e])
    a = 1.0
    b = 2.0
    demo = Demo(a=a, b=b)

    expected_c = 3.0
    expected_d = 4.0
    expected_e = 7.0
    with backend, basic:
        assert demo.c == expected_c
        assert demo.d == expected_d
        assert demo.e == expected_e


def test_evaluation_of_attribute_with_no_path_raises_system_evaluation_error() -> None:
    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()
        c: float = field()

    @transform
    def d(a: float, b: float) -> float:
        return a + b

    basic = Implementation(Demo, [d])
    a = 1.0
    b = 2.0
    demo = Demo(a=a, b=b)

    with basic, pytest.raises(BiltAnalysisError):
        _ = demo.c


def test_evaluation_with_single_equation() -> None:
    @system
    class NewtonsSecondLaw(metaclass=Interface):
        f: float = field()
        m: float = field()
        a: float = field()

    @equation
    def force_law(f: float, m: float, a: float) -> bool:
        return f == m * a

    impl = Implementation(NewtonsSecondLaw, [force_law])
    instance = NewtonsSecondLaw(m=2.0, a=3.0)

    expected_f = 2.0 * 3.0

    with impl:
        assert instance.f == expected_f


def test_evaluation_with_single_equation_solve_for_different_target() -> None:
    @system
    class NewtonsSecondLaw(metaclass=Interface):
        f: float = field()
        m: float = field()
        a: float = field()

    @equation
    def force_law(f: float, m: float, a: float) -> bool:
        return f == m * a

    impl = Implementation(NewtonsSecondLaw, [force_law])
    instance = NewtonsSecondLaw(f=6.0, m=2.0)

    expected_a = 6.0 / 2.0

    with impl:
        assert instance.a == expected_a


@pytest.mark.parametrize(
    ("backend"),
    (
        PYTHON_BACKEND,
        JAX_BACKEND,
    ),
)
def test_evaluation_with_equation_and_backend(backend: Backend) -> None:
    @system
    class NewtonsSecondLaw(metaclass=Interface):
        f: float = field()
        m: float = field()
        a: float = field()

    @equation
    def force_law(f: float, m: float, a: float) -> bool:
        return f == m * a

    impl = Implementation(NewtonsSecondLaw, [force_law])
    instance = NewtonsSecondLaw(m=2.0, a=3.0)

    expected_f = 3.0 * 2.0
    with backend, impl:
        assert instance.f == expected_f


def test_evaluation_with_transform_then_equation() -> None:
    """Transform computes an intermediate that is consumed by the equation."""

    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()
        d: float = field()

    @transform
    def c(a: float, b: float) -> float:
        return a + b

    @equation
    def sum_eq(a: float, c: float, d: float) -> bool:
        return d == a + c

    impl = Implementation(Demo, [c, sum_eq])
    instance = Demo(a=1.0, b=2.0)

    expected_d = 1.0 + (1.0 + 2.0)

    with impl:
        assert instance.d == expected_d


def test_evaluation_with_equation_then_transform() -> None:
    """Equation computes an intermediate that is consumed by the transform."""

    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()
        d: float = field()

    @equation
    def sum_eq(a: float, b: float, c: float) -> bool:
        return c == a + b

    @transform
    def d(a: float, c: float) -> float:
        return a * c

    impl = Implementation(Demo, [sum_eq, d])
    instance = Demo(a=2.0, b=3.0)

    expected_d = (2.0 + 3.0) * 2.0
    with impl:
        assert instance.d == expected_d


@pytest.mark.parametrize(
    ("inputs", "output", "expected"),
    (
        ({"i": 2.0, "r": 3.0}, "v", 6.0),
        ({"v": 6.0, "r": 3.0}, "i", 2.0),
        ({"v": 6.0, "i": 2.0}, "r", 3.0),
        ({"v": 6.0, "i": 2.0}, "p", 12.0),
        ({"p": 12.0, "v": 6.0}, "i", 2.0),
        ({"p": 12.0, "i": 2.0}, "r", 3.0),
        ({"p": 12.0, "i": 2.0}, "v", 6.0),
    ),
)
def test_evaluation_with_multiple_equations_many_directions(
    inputs: dict[str, float], output: str, expected: float
) -> None:
    @system
    class ElectricalCircuit(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()
        p: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    @equation
    def power_law(p: float, v: float, i: float) -> bool:
        return p == v * i

    impl = Implementation(ElectricalCircuit, [ohms_law, power_law])
    instance = ElectricalCircuit(**inputs)

    with impl:
        assert getattr(instance, output) == expected


@pytest.mark.parametrize(
    ("backend_context"),
    (
        contextlib.nullcontext(),
        PYTHON_BACKEND,
    ),
)
def test_repeated_evaluation_reuses_one_cache_entry(
    backend_context: contextlib.AbstractContextManager[object],
) -> None:
    """Repeated evaluation collapses to one cache entry, with or without an explicit backend context.

    Each backend is a single shared instance, so entering that instance on every iteration - or
    falling back to the default backend - keys the same cache entry every time.
    """

    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()
        c: float = field()

    @transform
    def c(a: float, b: float) -> float:
        return a + b

    implementation = Implementation(Demo, [c])

    with implementation:
        for value in range(20):
            with backend_context:
                assert Demo(a=float(value), b=2.0).c == value + 2.0

    assert len(implementation._evaluation_cache) == 1  # noqa: SLF001


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize("order", (("x", "y"), ("y", "x")), ids=("x_first", "y_first"))
@pytest.mark.parametrize("batched", (False, True), ids=("scalar", "tensor"))
@pytest.mark.parametrize("nested", (False, True), ids=("flat", "child"))
def test_default_causalization_keeps_supplied_inputs_separate_from_cached_outputs(
    backend: Backend, order: tuple[str, str], *, batched: bool, nested: bool
) -> None:
    @system
    class Pair(metaclass=Interface):
        x: float = field()
        y: float = field()
        p: float | Tensor = field()
        q: float = field()

    @equation
    def total(x: float, y: float, p: float) -> bool:
        return x + y == p

    @equation
    def difference(x: float, y: float, q: float) -> bool:
        return x - y == q

    @equation
    def supply_p(p: float, inner: Pair) -> bool:
        return inner.p == p

    @equation
    def supply_q(q: float, inner: Pair) -> bool:
        return inner.q == q

    @equation
    def expose_x(x: float, inner: Pair) -> bool:
        return x == inner.x

    @equation
    def expose_y(y: float, inner: Pair) -> bool:
        return y == inner.y

    model = (
        Implementation(
            Pair,
            [supply_p, supply_q, expose_x, expose_y],
            internal_implementations={"inner": Implementation(Pair, [total, difference])},
        )
        if nested
        else Implementation(Pair, [total, difference])
    )
    axis = dim("operating_point")
    p = tensor([10.0, 14.0], over=axis) if batched else 10.0
    expected = {"x": [6.0, 8.0] if batched else 6.0, "y": [4.0, 6.0] if batched else 4.0}
    with backend, model:
        instance = Pair(p=p, q=2.0)
        for name in order:
            result = getattr(instance, name)
            if batched:
                assert isinstance(result, Tensor)
                assert result.domain.dims == (axis,)
                np.testing.assert_allclose(result.values, expected[name], atol=1e-5)
            else:
                assert result == pytest.approx(expected[name], abs=1e-5)
            assert getattr(instance, name) is result
