__all__ = [
    "EvaluationContext",
    "current_evaluation",
]

import functools
import typing

from bilt.backend._base_backend import current_backend
from bilt.core._canonical_ordering import _derive_canonical_order
from bilt.core._domain_registry import DomainRegistry
from bilt.core._errors import BiltNoActiveEvaluationError

if typing.TYPE_CHECKING:
    from bilt.backend._base_backend import Backend
    from bilt.symbolic._domains import Domain

_evaluation_stack: list[EvaluationContext] = []


@typing.final
class EvaluationContext:
    """State shared across one Bilt evaluation, including the backend it compiles under.

    The backend is fixed when the evaluation is entered and never changes while it is in
    force, so everything one evaluation compiles is compiled by one backend.

    Entering under the backend already in force joins the active evaluation, which is how a
    sub-system shares its parent's domain bookkeeping. Entering under a different backend
    starts a separate evaluation instead, which is what lets a transform running under one
    backend evaluate another implementation under another.

    An evaluation also activates its backend for as long as it is in force, so the backend a
    nested evaluation resolves is the one its parent is using unless something says otherwise.
    """

    backend: Backend

    def __init__(self, backend: Backend | None = None) -> None:
        self.domains = DomainRegistry()
        self._requested_backend = backend

    @functools.cached_property
    def canonical_order(self) -> Domain:
        self.domains.lock_against_new_dimensions()
        return _derive_canonical_order(self)

    def __enter__(self) -> typing.Self:
        backend = current_backend() if self._requested_backend is None else self._requested_backend
        active = _evaluation_stack[-1] if _evaluation_stack else None

        if active is not None and (active is self or active.backend is backend):
            evaluation = active
            backend = active.backend
        else:
            evaluation = self

        # set even when joining, so this instance reports the backend actually in force
        self.backend = backend

        _evaluation_stack.append(evaluation)
        backend.__enter__()
        return typing.cast("typing.Self", evaluation)

    def __exit__(self, *_: object) -> None:
        _evaluation_stack.pop()
        self.backend.__exit__(None, None, None)


def current_evaluation() -> EvaluationContext:
    """Return the currently active evaluation."""
    if not _evaluation_stack:
        message = "No Bilt evaluation is currently active."
        raise BiltNoActiveEvaluationError(message)

    return _evaluation_stack[-1]
