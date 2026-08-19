import functools

__all__ = [
    "EvaluationContext",
    "current_evaluation",
]

import typing

from bilt.core._domain_registry import DomainRegistry
from bilt.symbolic._domains import Domain

_evaluation_stack: list[EvaluationContext] = []


@typing.final
class EvaluationContext:
    """State shared across one Bilt evaluation."""

    def __init__(self) -> None:
        self.domains = DomainRegistry()

    @functools.cached_property
    def canonical_order(self) -> Domain:
        self.domains.lock_against_new_dimensions()
        return Domain(tuple(current_evaluation().domains._bindings.keys()))  # noqa: SLF001

    def __enter__(self) -> typing.Self:
        evaluation = _evaluation_stack[-1] if _evaluation_stack else self
        _evaluation_stack.append(evaluation)
        return typing.cast("typing.Self", evaluation)

    def __exit__(self, *_: object) -> None:
        _evaluation_stack.pop()


def current_evaluation() -> EvaluationContext:
    """Return the currently active evaluation."""
    if not _evaluation_stack:
        message = "No Bilt evaluation is currently active."
        raise RuntimeError(message)

    return _evaluation_stack[-1]
