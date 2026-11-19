"""Request-local cache of backend-neutral program schedules.

A cache scope is shared by nested structural analyses and discarded when the
outermost request completes. Cache entries contain no backend callables or
request-specific numerical domains.
"""

import contextlib
import contextvars
import typing

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._program_ir import ProgramIR


type ProgramCache = dict[tuple[object, ...], ProgramIR]

_cache: contextvars.ContextVar[ProgramCache | None] = contextvars.ContextVar("bilt_program_ir_cache", default=None)


def get_program_cache() -> ProgramCache | None:
    """Return the current request's cache, or None for standalone realisation."""
    return _cache.get()


@contextlib.contextmanager
def program_cache_scope() -> collections.abc.Generator[ProgramCache]:
    """Reuse identical schedules in nested analysis, without cross-request state."""
    if (existing := _cache.get()) is not None:
        yield existing
        return
    cache: ProgramCache = {}
    token = _cache.set(cache)
    try:
        yield cache
    finally:
        _cache.reset(token)
