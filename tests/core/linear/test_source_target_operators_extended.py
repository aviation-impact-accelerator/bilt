"""Stress tests for the draft source-to-target linear-operator prototype."""

import typing

import jax
import numpy as np
import pytest

from bilt import (
    Implementation,
    Interface,
    Quantity,
    QuantityTensor,
    Tensor,
    dim,
    domain,
    equation,
    field,
    system,
    tensor,
)
from bilt.errors import BiltLinearSolveError
from bilt.units.dimensions import LENGTH, TIME
from bilt.units.units import kilometer, meter, second

SOURCE = dim("source")
TARGET = dim("target")

Influence = typing.Annotated[Tensor, domain(TARGET, SOURCE)]
Circulation = typing.Annotated[Tensor, domain(SOURCE)]
Contribution = typing.Annotated[Tensor, domain(TARGET, SOURCE)]
Velocity = typing.Annotated[Tensor, domain(TARGET)]


@system
class LiftingLine(metaclass=Interface):
    """Minimal named tensor source-to-target system."""

    influence: Influence = field()
    circulation: Circulation = field()
    contribution: Contribution = field()
    velocity: Velocity = field()


@equation
def product(influence: Influence, circulation: Circulation, contribution: Contribution) -> bool:
    """Generate the per-source contribution."""
    return contribution == influence * circulation


@equation
def reduction(contribution: Contribution, velocity: Velocity) -> bool:
    """Sum source contributions to each target."""
    return contribution.sum(SOURCE) == velocity


@equation
def direct(influence: Influence, circulation: Circulation, velocity: Velocity) -> bool:
    """Direct equivalent of the two equations."""
    return (influence * circulation).sum(SOURCE) == velocity


if typing.TYPE_CHECKING:
    from bilt._backend import Backend


@pytest.mark.parametrize("n", (1, 3, 7))
@pytest.mark.parametrize("split", (False, True))
def test_larger_dense_operators_match_numpy(backend: Backend, n: int, *, split: bool) -> None:
    """The solve must not be special-cased to two source and target panels."""
    a = 4.0 * np.eye(n) + 0.3 * np.ones((n, n))
    expected = 0.25 * np.arange(1, n + 1)
    b = a @ expected
    with backend, Implementation(LiftingLine, [product, reduction] if split else [direct]):
        result = LiftingLine(
            influence=tensor(a, over=(TARGET, SOURCE)),
            velocity=tensor(b, over=TARGET),
        ).circulation
    assert isinstance(result, Tensor)
    assert result.domain == domain(SOURCE)
    np.testing.assert_allclose(result.values, expected, rtol=1e-5, atol=1e-6)


ReversedInfluence = typing.Annotated[Tensor, domain(SOURCE, TARGET)]


@equation
def reverse_axis_operator(
    influence: ReversedInfluence,
    circulation: typing.Annotated[Tensor, domain(SOURCE)],
    velocity: typing.Annotated[Tensor, domain(TARGET)],
) -> bool:
    """The physical ordering of matrix axes should not change the solution."""
    return (influence * circulation).sum(SOURCE) == velocity


def test_reversed_matrix_axis_order(backend: Backend) -> None:
    """The source axis need not be the final stored axis of influence."""
    a = np.array([[3.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 5.0]])
    expected = np.array([1.5, 2.0, 0.5])
    with backend, Implementation(LiftingLine, [reverse_axis_operator]):
        result = LiftingLine(
            influence=tensor(a.T, over=(SOURCE, TARGET)),
            velocity=tensor(a @ expected, over=TARGET),
        ).circulation
    assert isinstance(result, Tensor)
    assert result.domain == domain(SOURCE)
    np.testing.assert_allclose(result.values, expected, rtol=1e-5, atol=1e-6)


CASE = dim("flight_case")
CaseInfluence = typing.Annotated[Tensor, domain(CASE, TARGET, SOURCE)]
CaseCirculation = typing.Annotated[Tensor, domain(CASE, SOURCE)]
CaseVelocity = typing.Annotated[Tensor, domain(CASE, TARGET)]


@system
class BatchedMap(metaclass=Interface):
    """Independent matrix equations across operating cases."""

    influence: CaseInfluence = field()
    circulation: CaseCirculation = field()
    velocity: CaseVelocity = field()


@equation
def batch_direct(influence: CaseInfluence, circulation: CaseCirculation, velocity: CaseVelocity) -> bool:
    """Batch is retained when source is reduced and target is preserved."""
    return (influence * circulation).sum(SOURCE) == velocity


def test_independent_cases_are_not_mixed(backend: Backend) -> None:
    """The operator should be block diagonal across the named case axis."""
    matrices = np.array(
        [
            [[2.0, 1.0], [1.0, 3.0]],
            [[4.0, -1.0], [0.5, 2.0]],
            [[7.0, 2.0], [3.0, 5.0]],
        ]
    )
    expected = np.array([[1.6, 1.8], [2.0, 0.5], [-1.0, 2.0]])
    rhs = np.einsum("cij,cj->ci", matrices, expected)
    with backend, Implementation(BatchedMap, [batch_direct]):
        result = BatchedMap(
            influence=tensor(matrices, over=(CASE, TARGET, SOURCE)),
            velocity=tensor(rhs, over=(CASE, TARGET)),
        ).circulation
    assert isinstance(result, Tensor)
    assert result.domain == domain(CASE, SOURCE)
    np.testing.assert_allclose(result.values, expected, rtol=1e-5, atol=1e-6)


InputInfluence = typing.Annotated[QuantityTensor, LENGTH**-1, domain(TARGET, SOURCE)]
InputCirculation = typing.Annotated[QuantityTensor, LENGTH**2 / TIME, domain(SOURCE)]
InputVelocity = typing.Annotated[QuantityTensor, LENGTH / TIME, domain(TARGET)]


@system
class DimensionalMap(metaclass=Interface):
    """Dimensioned counterpart of the unitless operator."""

    influence: InputInfluence = field()
    circulation: InputCirculation = field()
    velocity: InputVelocity = field()


@equation
def dimensional_direct(
    influence: InputInfluence,
    circulation: typing.Annotated[Quantity | QuantityTensor, LENGTH**2 / TIME, domain(SOURCE)],
    velocity: InputVelocity,
) -> bool:
    """Physical units need coherent conversion during matrix lowering."""
    return (influence * circulation).sum(SOURCE) == velocity


def test_dimensioned_operator_preserves_units(backend: Backend) -> None:
    """Use reciprocal kilometres and kilometre/s input to obtain m²/s circulation."""
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    expected = np.array([1.6, 1.8])
    # Numerically: 1/km times (km²/s) gives km/s.
    with backend, Implementation(DimensionalMap, [dimensional_direct]):
        result = DimensionalMap(
            influence=tensor(a, over=(TARGET, SOURCE), unit=kilometer**-1),
            velocity=tensor(a @ expected, over=TARGET, unit=kilometer / second),
        ).circulation
    assert isinstance(result, QuantityTensor)
    assert result.domain == domain(SOURCE)
    np.testing.assert_allclose(
        result.convert_to(meter**2 / second).values,
        expected * 1e6,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("split", (False, True))
def test_singular_operator_is_rejected(backend: Backend, *, split: bool) -> None:
    """A mathematically singular square system must not return arbitrary values."""
    with (
        backend,
        Implementation(LiftingLine, [product, reduction] if split else [direct]),
        pytest.raises((BiltLinearSolveError, jax.errors.JaxRuntimeError)),
    ):
        _ = LiftingLine(
            influence=tensor([[1.0, 2.0], [2.0, 4.0]], over=(TARGET, SOURCE)),
            velocity=tensor([3.0, 6.0], over=TARGET),
        ).circulation


@system
class TrimAugmented(metaclass=Interface):
    """A source-to-target map coupled to a global scalar unknown."""

    influence: Tensor = field()
    circulation: Tensor = field()
    angle: float = field()
    velocity: Tensor = field()
    total: float = field()


@equation
def velocity_with_angle(influence: Tensor, circulation: Tensor, angle: float, velocity: Tensor) -> bool:
    """Every target depends on circulation and a common angle."""
    return (influence * circulation).sum(SOURCE) + angle == velocity


@equation
def fix_total_circulation(circulation: Tensor, total: float) -> bool:
    """The extra integral condition fixes the global degree of freedom."""
    return circulation.sum(SOURCE) == total


def test_mixed_scalar_and_source_tensor_unknowns(backend: Backend) -> None:
    """Two target equations plus one integral solve two circulations and one angle."""
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    expected_gamma = np.array([1.6, 1.8])
    expected_angle = 0.4
    rhs = a @ expected_gamma + expected_angle
    with backend, Implementation(TrimAugmented, [velocity_with_angle, fix_total_circulation]):
        result = TrimAugmented(
            influence=tensor(a, over=(TARGET, SOURCE)),
            velocity=tensor(rhs, over=TARGET),
            total=float(expected_gamma.sum()),
        ).circulation
    assert isinstance(result, Tensor)
    assert result.domain == domain(SOURCE)
    np.testing.assert_allclose(result.values, expected_gamma, rtol=1e-5, atol=1e-6)


# Keep the explicit-hint variant as a separate regression alongside automatic
# inference of mixed tensor and scalar unknowns.
SourceCirculation = typing.Annotated[Tensor, domain(SOURCE)]
SharedAngle = typing.Annotated[float, domain()]


@equation
def velocity_with_angle_hinted(
    influence: Tensor,
    circulation: SourceCirculation,
    angle: SharedAngle,
    velocity: Tensor,
) -> bool:
    """Map source circulation to target velocity plus a shared scalar."""
    return (influence * circulation).sum(SOURCE) + angle == velocity


@equation
def fix_total_circulation_hinted(circulation: SourceCirculation, total: float) -> bool:
    """A scalar equation closes the mixed-layout system."""
    return circulation.sum(SOURCE) == total


def test_mixed_source_tensor_and_shared_scalar_with_exact_hints(backend: Backend) -> None:
    """Check whether mixed assembly already works once the domains are explicit."""
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    expected_gamma = np.array([1.6, 1.8])
    expected_angle = 0.4
    rhs = a @ expected_gamma + expected_angle
    with backend, Implementation(TrimAugmented, [velocity_with_angle_hinted, fix_total_circulation_hinted]):
        result = TrimAugmented(
            influence=tensor(a, over=(TARGET, SOURCE)),
            velocity=tensor(rhs, over=TARGET),
            total=float(expected_gamma.sum()),
        ).circulation

    assert isinstance(result, Tensor)
    assert result.domain == domain(SOURCE)
    np.testing.assert_allclose(result.values, expected_gamma, rtol=1e-5, atol=1e-6)
