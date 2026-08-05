__all__ = [
    "EvaluationContext",
    "current_evaluation",
]

import contextvars
import typing

from bilt.core._domain_registry import DomainRegistry

_ACTIVE_EVALUATION: contextvars.ContextVar[EvaluationContext | None] = contextvars.ContextVar(
    "_ACTIVE_EVALUATION",
    default=None,
)


@typing.final
class EvaluationContext:
    """State shared across one Bilt evaluation."""

    def __init__(self) -> None:
        self.domains = DomainRegistry()
        self._tokens: list[contextvars.Token[EvaluationContext | None] | None] = []

    def __enter__(self) -> typing.Self:
        active = _ACTIVE_EVALUATION.get()

        if active is not None:
            self._tokens.append(None)
            return typing.cast("typing.Self", active)

        token = _ACTIVE_EVALUATION.set(self)
        self._tokens.append(token)
        return self

    def __exit__(self, *_: object) -> None:
        token = self._tokens.pop()

        if token is not None:
            _ACTIVE_EVALUATION.reset(token)


def current_evaluation() -> EvaluationContext:
    """Return the currently active evaluation."""
    evaluation = _ACTIVE_EVALUATION.get()

    if evaluation is None:
        message = "No Bilt evaluation is currently active."
        raise RuntimeError(message)

    return evaluation
