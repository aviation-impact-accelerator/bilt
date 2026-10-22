"""Reuse local structural inspections within one nested analysis request."""

__all__ = ["get_structural_cache", "structural_cache_scope"]

import contextlib
import contextvars
import typing

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
    from bilt.core._structural_analysis._causalization_block_solver._structural_scope import StructuralScope


type StructuralCache = dict[tuple[FullIncidenceGraph, frozenset[str], frozenset[str]], StructuralScope]

_cache: contextvars.ContextVar[StructuralCache | None] = contextvars.ContextVar("bilt_structural_cache", default=None)


def get_structural_cache() -> StructuralCache | None:
    """Return this request's cache, or None outside an analysis scope."""
    return _cache.get()


@contextlib.contextmanager
def structural_cache_scope() -> collections.abc.Generator[StructuralCache]:
    """Share structural evidence with nested calls and reset after the outer call.

    Structural results do not depend on execution failure knowledge. Route-cache
    invalidation therefore leaves this cache intact until the request ends.
    """
    existing = _cache.get()
    if existing is not None:
        yield existing
        return

    cache: StructuralCache = {}
    token = _cache.set(cache)
    try:
        yield cache
    finally:
        _cache.reset(token)
