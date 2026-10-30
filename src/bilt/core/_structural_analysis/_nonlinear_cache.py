"""Instance-owned complete nonlinear solutions, shared across output requests."""

import contextlib
import contextvars
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Generator

    from bilt.core._structural_analysis._qualified_block import QualifiedEquation


type NonlinearSolutionKey = tuple[
    tuple[str, ...], object, frozenset[QualifiedEquation], frozenset[str], tuple[tuple[str, float], ...]
]
type NonlinearSolutions = dict[NonlinearSolutionKey, dict[str, float]]

_cache: contextvars.ContextVar[NonlinearSolutions | None] = contextvars.ContextVar("bilt_nonlinear_cache", default=None)
_path: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("bilt_nonlinear_path", default=())


def current_nonlinear_cache() -> tuple[NonlinearSolutions | None, tuple[str, ...]]:
    """Return the current instance's solutions and owning subsystem namespace."""
    return _cache.get(), _path.get()


@contextlib.contextmanager
def using_nonlinear_cache(solutions: NonlinearSolutions) -> Generator[None]:
    """Scope an instance cache to one attribute request, restoring nested callers."""
    token = _cache.set(solutions)
    path_token = _path.set(())
    try:
        yield
    finally:
        _path.reset(path_token)
        _cache.reset(token)


@contextlib.contextmanager
def nonlinear_subsystem(name: str) -> Generator[None]:
    """Keep sibling instances distinct even when they reuse equations and options."""
    token = _path.set((*_path.get(), name))
    try:
        yield
    finally:
        _path.reset(token)
