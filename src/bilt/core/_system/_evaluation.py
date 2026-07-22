from __future__ import annotations

__all__: list[str] = []

import collections.abc
import contextlib
import contextvars

from bilt.core._system._domain_registry import DomainRegistry


_ACTIVE_DOMAIN_REGISTRY: contextvars.ContextVar[
    DomainRegistry | None
] = contextvars.ContextVar(
    "_ACTIVE_DOMAIN_REGISTRY",
    default=None,
)


@contextlib.contextmanager
def domain_registry_scope(
) -> collections.abc.Iterator[DomainRegistry]:
    """Create or reuse the registry for the current evaluation."""

    active = _ACTIVE_DOMAIN_REGISTRY.get()

    if active is not None:
        yield active
        return

    registry = DomainRegistry()
    token = _ACTIVE_DOMAIN_REGISTRY.set(registry)

    try:
        yield registry
    finally:
        _ACTIVE_DOMAIN_REGISTRY.reset(token)


def _active_domain_registry() -> DomainRegistry:
    """Return the registry for internal evaluation code."""

    registry = _ACTIVE_DOMAIN_REGISTRY.get()

    if registry is None:
        message = "No Bilt evaluation is currently active."
        raise RuntimeError(message)

    return registry
