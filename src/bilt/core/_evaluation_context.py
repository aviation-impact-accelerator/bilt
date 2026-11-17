from bilt.core._canonical_ordering import _derive_canonical_order
from bilt.core._errors import BiltNoActiveEvaluationError

__all__ = [
    "EvaluationContext",
    "current_evaluation",
]

import typing

from bilt.core._domain_registry import DomainRegistry

if typing.TYPE_CHECKING:
    from bilt.symbolic._domains import Domain

_evaluation_stack: list[EvaluationContext] = []


@typing.final
class EvaluationContext:
    """State shared across one Bilt evaluation."""

    def __init__(self) -> None:
        self.domains = DomainRegistry()
        self._canonical_order: Domain | None = None

    @property
    def canonical_order(self) -> Domain:
        """Freeze and return the canonical order for executable operations."""
        if self._canonical_order is None:
            self.domains.lock_against_new_dimensions()
            self._canonical_order = _derive_canonical_order(self)
        return self._canonical_order

    @canonical_order.setter
    def canonical_order(self, order: Domain) -> None:
        """Allow an explicit canonical order, including test-time overrides."""
        self._canonical_order = order

    def canonical_order_for_cache(self) -> Domain:
        """Inspect the current order without preventing derived axes from binding."""
        if self._canonical_order is not None:
            return self._canonical_order
        return self.domains.dimension_order()

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
        raise BiltNoActiveEvaluationError(message)

    return _evaluation_stack[-1]
