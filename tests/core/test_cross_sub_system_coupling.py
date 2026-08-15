"""Acceptance tests for coupled solves that cross sub-system boundaries."""

import pytest

from bilt.core import (
    Implementation,
    Interface,
    field,
    system,
    transform,
)
from bilt.core._equation import equation
from bilt._backend._jax import JAXBackend
from bilt._backend._python import PythonBackend

type Backend = PythonBackend | JAXBackend


@system
class Spring(metaclass=Interface):
    force: float = field()
    extension: float = field()
    stiffness: float = field()


@equation
def hookes_law(
    force: float,
    extension: float,
    stiffness: float,
) -> bool:
    return force == stiffness * extension


spring_impl = Implementation(
    Spring,
    [hookes_law],
)


@system
class DoubleSpring(metaclass=Interface):
    force: float = field()
    total_extension: float = field()
    first_stiffness: float = field()
    second_stiffness: float = field()
    first_extension: float = field()
    second_extension: float = field()
    stored_energy: float = field()


@equation
def first_force(
    force: float,
    first: Spring,
) -> bool:
    return force == first.force


@equation
def second_force(
    force: float,
    second: Spring,
) -> bool:
    return force == second.force


@equation
def extensions(
    total_extension: float,
    first: Spring,
    second: Spring,
) -> bool:
    return total_extension == first.extension + second.extension


@equation
def first_stiffness(
    first_stiffness: float,
    first: Spring,
) -> bool:
    return first_stiffness == first.stiffness


@equation
def second_stiffness(
    second_stiffness: float,
    second: Spring,
) -> bool:
    return second_stiffness == second.stiffness


@equation
def expose_first_extension(
    first_extension: float,
    first: Spring,
) -> bool:
    return first_extension == first.extension


@equation
def expose_second_extension(
    second_extension: float,
    second: Spring,
) -> bool:
    return second_extension == second.extension


@transform
def stored_energy(
    force: float,
    total_extension: float,
) -> float:
    return 0.5 * force * total_extension


double_spring_impl = Implementation(
    DoubleSpring,
    [
        first_force,
        second_force,
        extensions,
        first_stiffness,
        second_stiffness,
        expose_first_extension,
        expose_second_extension,
        stored_energy,
    ],
    internal_implementations={
        "first": spring_impl,
        "second": spring_impl,
    },
)


@pytest.mark.parametrize(
    ("total_extension", "first_stiffness", "second_stiffness", "expected_force"),
    (
        (0.1, 100.0, 300.0, 7.5),
        (0.1, 200.0, 200.0, 10.0),
        (0.3, 50.0, 100.0, 10.0),
    ),
)
@pytest.mark.parametrize(
    "backend",
    (PythonBackend(), JAXBackend()),
)
def test_series_springs_solve_shared_force(
    backend: Backend,
    total_extension: float,
    first_stiffness: float,
    second_stiffness: float,
    expected_force: float,
) -> None:
    """The parent and both child springs participate in one coupled solve."""
    instance = DoubleSpring(
        total_extension=total_extension,
        first_stiffness=first_stiffness,
        second_stiffness=second_stiffness,
    )

    with backend, double_spring_impl:
        assert instance.force == pytest.approx(expected_force)


@pytest.mark.parametrize(
    "backend",
    (PythonBackend(), JAXBackend()),
)
def test_series_springs_expose_values_solved_inside_children(
    backend: Backend,
) -> None:
    """Values produced inside each child are available after the coupled solve."""
    instance = DoubleSpring(
        total_extension=0.1,
        first_stiffness=100.0,
        second_stiffness=300.0,
    )

    with backend, double_spring_impl:
        assert instance.first_extension == pytest.approx(0.075)
        assert instance.second_extension == pytest.approx(0.025)


@pytest.mark.parametrize(
    "backend",
    (PythonBackend(), JAXBackend()),
)
def test_series_springs_coupled_result_is_consumed_by_outer_transform(
    backend: Backend,
) -> None:
    """A value from the cross-boundary coupled solve feeds later parent work."""
    instance = DoubleSpring(
        total_extension=0.1,
        first_stiffness=100.0,
        second_stiffness=300.0,
    )

    with backend, double_spring_impl:
        assert instance.stored_energy == pytest.approx(0.375)
