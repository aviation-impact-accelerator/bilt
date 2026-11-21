"""Prototype inverse source-to-target linear operators using named dimensions."""

import typing

import numpy as np
import pytest

from bilt import Implementation, Interface, Tensor, dim, domain, equation, field, system, tensor
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.errors import BiltLinearSolveError

if typing.TYPE_CHECKING:
    from bilt._backend import Backend

SOURCE = dim("source")
TARGET = dim("target")

Influence = typing.Annotated[Tensor, domain(TARGET, SOURCE)]
Circulation = typing.Annotated[Tensor, domain(SOURCE)]
Contribution = typing.Annotated[Tensor, domain(TARGET, SOURCE)]
Velocity = typing.Annotated[Tensor, domain(TARGET)]


@system
class LiftingLine(metaclass=Interface):
    """A prescribed linear influence map from source to target panels."""

    influence: Influence = field()
    circulation: Circulation = field()
    contribution: Contribution = field()
    velocity: Velocity = field()


@equation
def product(influence: Influence, circulation: Circulation, contribution: Contribution) -> bool:
    """The individual contribution at each target from each source."""
    return contribution == influence * circulation


@equation
def reduction(contribution: Contribution, velocity: Velocity) -> bool:
    """The net normal velocity at each target."""
    return contribution.sum(SOURCE) == velocity


@equation
def direct(influence: Influence, circulation: Circulation, velocity: Velocity) -> bool:
    """The equivalent single-equation contraction."""
    return (influence * circulation).sum(SOURCE) == velocity


@pytest.mark.parametrize("split", (False, True))
def test_inverse_source_to_target_linear_operator(backend: Backend, *, split: bool) -> None:
    """Infer circulation by solving the non-pointwise operator, not inverting the sum."""
    influence = tensor([[2.0, 1.0], [1.0, 3.0]], over=(TARGET, SOURCE))
    velocity = tensor([5.0, 7.0], over=TARGET)
    implementation = Implementation(LiftingLine, [product, reduction] if split else [direct])

    with backend, implementation:
        model = LiftingLine(influence=influence, velocity=velocity)
        circulation = model.circulation
    assert isinstance(circulation, Tensor)
    assert circulation.domain == domain(SOURCE)
    np.testing.assert_allclose(circulation.values, np.linalg.solve(influence.values, velocity.values), atol=1e-6)
    if split:
        # Evaluate this separate request in the supported forward direction.
        # Re-querying the original inverse instance does not reuse the prior
        # solution and is a distinct multi-output planning limitation.
        with backend, implementation:
            followup = LiftingLine(influence=influence, circulation=circulation)
            contribution = followup.contribution
        assert isinstance(contribution, Tensor)
        assert contribution.domain == domain(TARGET, SOURCE)
        np.testing.assert_allclose(contribution.values, [[3.2, 1.8], [1.6, 5.4]], atol=1e-6)


def test_two_step_forward_evaluation_uses_plain_tensor_operations(backend: Backend) -> None:
    """The special inverse lowering must not disturb normal forward evaluation."""
    with backend, Implementation(LiftingLine, [product, reduction]):
        model = LiftingLine(
            influence=tensor([[2.0, 1.0], [1.0, 3.0]], over=(TARGET, SOURCE)),
            circulation=tensor([1.6, 1.8], over=SOURCE),
        )
        contribution = model.contribution
        velocity = model.velocity

    assert isinstance(contribution, Tensor)
    assert isinstance(velocity, Tensor)
    np.testing.assert_allclose(contribution.values, [[3.2, 1.8], [1.6, 5.4]], atol=1e-6)
    np.testing.assert_allclose(velocity.values, [5.0, 7.0], atol=1e-6)


@pytest.mark.parametrize("split", (False, True))
def test_non_square_source_to_target_system_is_rejected(backend: Backend, *, split: bool) -> None:
    """A 2x3 operator cannot uniquely recover its three circulation variables."""
    influence = tensor([[2.0, 1.0, 0.5], [1.0, 3.0, 0.5]], over=(TARGET, SOURCE))
    velocity = tensor([5.0, 7.0], over=TARGET)
    with (
        backend,
        Implementation(LiftingLine, [product, reduction] if split else [direct]),
        pytest.raises((BiltExecutionPlanError, BiltLinearSolveError)),
    ):
        _ = LiftingLine(influence=influence, velocity=velocity).circulation
