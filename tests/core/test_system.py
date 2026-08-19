import contextlib
import typing
from typing import TYPE_CHECKING

import pytest

from bilt import (
    Implementation,
    Interface,
    field,
    system,
    transform,
)
from bilt._backend import BackendCallable
from bilt._backend._base_backend import ActivatableBackendMixin
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._ndarray_alignment_mixin import NDArrayAlignmentMixin
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.errors import (
    BiltAnalysisError,
    BiltSystemDefinitionError,
    BiltUnknownSystemAttributeError,
)

if TYPE_CHECKING:
    import ast

    from bilt._backend import BackendCase
    from bilt._backend._base_backend import Backend
    from bilt.core._system._instance import SystemInstance


def _instance_variables(instance: object) -> dict[str, object]:
    """The values an instance holds, keyed by name, including those with no declared field."""
    return typing.cast("SystemInstance", instance).__bilt_system_instance_variables__


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
    falling back to the default backend - keys the same cache entry every time. Each iteration also
    builds a fresh instance, so caching intermediates on the instance never widens the available set
    of a later request and so never adds a cache entry here.
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


@pytest.mark.parametrize(
    ("backend"),
    (
        PYTHON_BACKEND,
        JAX_BACKEND,
    ),
)
def test_evaluation_caches_undeclared_intermediates(backend: Backend) -> None:
    """Every value computed on the way to an attribute is cached, including undeclared ones."""

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

    expected_c = 1.0 + 2.0
    expected_d = 1.0 + (1.0 + 2.0)

    with backend, impl:
        assert instance.d == expected_d

    cached_c = typing.cast("float", _instance_variables(instance)["c"])
    assert float(cached_c) == pytest.approx(expected_c)


def test_evaluation_leaves_supplied_inputs_untouched() -> None:
    """Caching outputs never rebinds a value the caller supplied, so no `jax.Array` replaces their float."""

    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()
        c: float = field()

    @transform
    def c(a: float, b: float) -> float:
        return a + b

    impl = Implementation(Demo, [c])
    a_value = 1.0
    instance = Demo(a=a_value, b=2.0)

    with JAX_BACKEND, impl:
        _ = instance.c

    assert _instance_variables(instance)["a"] is a_value


@pytest.mark.parametrize(
    ("backend"),
    (
        PYTHON_BACKEND,
        JAX_BACKEND,
    ),
)
def test_cached_intermediates_are_available_to_later_evaluations(backend: Backend) -> None:
    """A cached intermediate joins the available set of the next request planned for the same instance.

    Reading `d` computes `c` on the way, so the plan for `e` is asked for from a wider set of
    available values than the one the instance was constructed with.
    """

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

    implementation = Implementation(Demo, [c, d, e])
    instance = Demo(a=1.0, b=2.0)

    with backend, implementation:
        _ = instance.d
        _ = instance.e

    requests = [request for request, *_ in implementation._evaluation_cache]  # noqa: SLF001
    assert any(request.output == "e" and "c" in request.available for request in requests)


def test_recomputing_an_available_value_raises_internal_error() -> None:
    """A plan that hands back a value the instance already holds is a bug in bilt, and says so."""

    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()
        c: float = field()

    @transform
    def c(a: float, b: float) -> float:
        return a + b

    class CollidingBackend(ActivatableBackendMixin, NDArrayAlignmentMixin):
        """Rigs every plan to also produce a value for `a`, which the instance already holds."""

        def transpile(self, function_definition: ast.FunctionDef) -> BackendCallable:
            return PYTHON_BACKEND.transpile(function_definition)

        def chain_case(self, case: BackendCase) -> BackendCallable:
            return BackendCallable(
                name=case.return_name,
                parameters=case.parameters,
                callable=lambda **_: {"c": 3.0, "a": 99.0},
                outputs=("c", "a"),
            )

    implementation = Implementation(Demo, [c])
    instance = Demo(a=1.0, b=2.0)

    with CollidingBackend(), implementation, pytest.raises(RuntimeError):
        _ = instance.c
