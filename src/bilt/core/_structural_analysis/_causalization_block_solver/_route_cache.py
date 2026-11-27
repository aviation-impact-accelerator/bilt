"""Reuse execution-route queries within one nested analysis call."""

__all__ = ["invalidate_route_cache", "route_cache_scope", "route_dependency_scope", "selective_route_cache_scope"]

import contextlib
import contextvars
import typing

if typing.TYPE_CHECKING:
    import collections.abc


type RouteKey = tuple[object, frozenset[str], str]
type RouteCache = dict[RouteKey, frozenset[str] | None]


class _RequestRouteCache(dict[RouteKey, frozenset[str] | None]):
    """Keep query dependencies only for the lifetime of one analysis request."""

    def __init__(self) -> None:
        super().__init__()
        self.dependencies: dict[object, set[object]] = {}
        self.active_owners: list[object] = []


_cache: contextvars.ContextVar[_RequestRouteCache | None] = contextvars.ContextVar("bilt_route_cache", default=None)
_selective: contextvars.ContextVar[bool] = contextvars.ContextVar("bilt_selective_route_cache", default=False)


@contextlib.contextmanager
def selective_route_cache_scope() -> collections.abc.Generator[None]:
    """Opt in to invalidating only routes depending on a failed analyser."""
    token = _selective.set(True)
    try:
        yield
    finally:
        _selective.reset(token)


@contextlib.contextmanager
def route_dependency_scope(owner: object) -> collections.abc.Generator[None]:
    """Record parent-to-child route queries, including hits in the route cache.

    Record dependencies at analyser granularity rather than by individual route:
    this may evict more entries than strictly necessary, but cannot preserve
    another route of a parent after one of its child analysers has changed.
    """
    cache = _cache.get()
    if not _selective.get() or cache is None:
        yield
        return

    if cache.active_owners:
        parent = cache.active_owners[-1]
        if parent is not owner:
            cache.dependencies.setdefault(parent, set()).add(owner)
    cache.active_owners.append(owner)
    try:
        yield
    finally:
        cache.active_owners.pop()


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

    cache = _RequestRouteCache()
    token = _cache.set(cache)
    try:
        yield cache
    finally:
        _cache.reset(token)


def invalidate_route_cache(owner: object | None = None) -> None:
    """Invalidate changed routes and their transitive parents.

    Without an owner, or outside the opt-in scope, preserve the existing
    conservative behaviour of discarding all route answers.
    """
    cache = _cache.get()
    if cache is None:
        return
    if not _selective.get() or owner is None:
        cache.clear()
        return

    affected = {owner}
    while True:
        parents = {
            parent
            for parent, children in cache.dependencies.items()
            if parent not in affected and not children.isdisjoint(affected)
        }
        if not parents:
            break
        affected.update(parents)
    for key in tuple(cache):
        if key[0] in affected:
            del cache[key]
