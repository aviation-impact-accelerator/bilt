import pytest

from bilt.backend._base_backend import current_backend
from bilt.backend._jax._backend import JAX_BACKEND
from bilt.backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import (
    EvaluationContext,
    current_evaluation,
)
from bilt.errors import BiltNoActiveEvaluationError


def test_current_evaluation_requires_active_context() -> None:
    with pytest.raises(
        BiltNoActiveEvaluationError,
        match="No Bilt evaluation is currently active",
    ):
        current_evaluation()


def test_nested_contexts_share_outer_evaluation() -> None:
    with EvaluationContext() as outer:
        assert current_evaluation() is outer

        with EvaluationContext() as nested:
            assert nested is outer
            assert current_evaluation() is outer
            assert nested.domains is outer.domains

        assert current_evaluation() is outer

    with pytest.raises(
        BiltNoActiveEvaluationError,
        match="No Bilt evaluation is currently active",
    ):
        current_evaluation()


def test_an_explicit_backend_is_the_one_in_force() -> None:
    with EvaluationContext(backend=JAX_BACKEND) as evaluation:
        assert evaluation.backend is JAX_BACKEND


def test_an_activated_backend_is_the_one_in_force() -> None:
    with JAX_BACKEND, EvaluationContext() as evaluation:
        assert evaluation.backend is JAX_BACKEND


def test_the_python_backend_is_in_force_when_none_is_activated() -> None:
    with EvaluationContext() as evaluation:
        assert evaluation.backend is PYTHON_BACKEND


def test_an_explicit_backend_beats_the_activation_stack() -> None:
    with PYTHON_BACKEND, EvaluationContext(backend=JAX_BACKEND) as evaluation:
        assert evaluation.backend is JAX_BACKEND


def test_an_evaluation_activates_its_backend_for_as_long_as_it_is_in_force() -> None:
    """A nested evaluation continues under its parent's backend rather than dropping to the default."""
    with EvaluationContext(backend=JAX_BACKEND) as outer:
        assert current_backend() is JAX_BACKEND

        with EvaluationContext() as nested:
            assert nested is outer

    assert current_backend() is PYTHON_BACKEND


def test_a_nested_context_under_the_same_backend_joins_the_active_evaluation() -> None:
    """Sharing is what lets a sub-system take part in its parent's domain bookkeeping."""
    with EvaluationContext(backend=JAX_BACKEND) as outer, EvaluationContext(backend=JAX_BACKEND) as nested:
        assert nested is outer
        assert nested.domains is outer.domains


def test_a_nested_context_naming_another_backend_is_a_separate_evaluation() -> None:
    """A transform may evaluate another implementation under another backend, so this is not refused."""
    with EvaluationContext(backend=PYTHON_BACKEND) as outer, EvaluationContext(backend=JAX_BACKEND) as nested:
        assert nested is not outer
        assert nested.backend is JAX_BACKEND
        assert nested.domains is not outer.domains


def test_the_outer_evaluation_is_restored_after_a_separate_nested_one() -> None:
    with EvaluationContext(backend=PYTHON_BACKEND) as outer:
        with EvaluationContext(backend=JAX_BACKEND):
            assert current_evaluation().backend is JAX_BACKEND

        assert current_evaluation() is outer
        assert outer.backend is PYTHON_BACKEND


def test_a_backend_activated_inside_an_evaluation_starts_a_separate_one() -> None:
    """This is the documented escape hatch: a Python transform delegating to a JAX implementation."""
    with EvaluationContext(backend=PYTHON_BACKEND) as outer:
        with JAX_BACKEND, EvaluationContext() as nested:
            assert nested is not outer
            assert nested.backend is JAX_BACKEND

        assert current_evaluation() is outer


def test_re_entering_an_active_evaluation_does_not_change_its_backend() -> None:
    """The backend of an evaluation in force is fixed, whatever is activated around it."""
    evaluation = EvaluationContext(backend=PYTHON_BACKEND)
    with evaluation, JAX_BACKEND, evaluation:
        assert evaluation.backend is PYTHON_BACKEND

    assert evaluation.backend is PYTHON_BACKEND
