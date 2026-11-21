"""Reuse execution-route queries within one nested analysis call."""

__all__ = ["invalidate_route_cache", "route_cache_scope"]

import contextlib
import contextvars
import typing

from bilt.core._structural_analysis._causalization_block_solver._settlement_cache import invalidate_settlement_cache

if typing.TYPE_CHECKING:
    import collections.abc


type RouteCache = dict[tuple[object, frozenset[str], str], frozenset[str] | None]

_cache: contextvars.ContextVar[RouteCache | None] = contextvars.ContextVar("bilt_route_cache", default=None)


@contextlib.contextmanager
def route_cache_scope() -> collections.abc.Generator[RouteCache]:
    """Share a cache with nested calls and discard it after the outermost call.

    Standalone route queries also establish a temporary scope. No entries survive
    into a later top-level analysis, even within the same EvaluationContext.
    """
    existing = _cache.get()
    if existing is not None:
        yield existing
        return

    cache: RouteCache = {}
    token = _cache.set(cache)
    try:
        yield cache
    finally:
        _cache.reset(token)


def invalidate_route_cache() -> None:
    """Discard route and settlement answers after realisation changes failure knowledge."""
    invalidate_settlement_cache()
    cache = _cache.get()
    if cache is not None:
        cache.clear()
