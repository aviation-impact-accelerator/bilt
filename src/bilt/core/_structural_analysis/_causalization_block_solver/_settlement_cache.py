"""Request-scoped, opt-in memoization of dependency settlement.

Full assignment identities are deliberately used here: narrowing the signature
to a dependency closure is a future optimisation requiring separate proofs.
"""

import contextlib
import contextvars
import typing

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._plan import PlanStep


type SettlementResult = tuple[tuple[PlanStep, ...], frozenset[str]]
type SettlementKey = tuple[int, str, frozenset[str], tuple[tuple[str, int], ...]]
type SettlementEntry = tuple[tuple[BiltModelConstituent, ...], SettlementResult]
type SettlementCache = dict[SettlementKey, SettlementEntry]

_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar("bilt_settlement_memo_enabled", default=False)
_cache: contextvars.ContextVar[SettlementCache | None] = contextvars.ContextVar(
    "bilt_settlement_memo_cache", default=None
)


@contextlib.contextmanager
def settlement_memo_scope() -> collections.abc.Generator[None]:
    """Opt in to exact settled-plan memoization, with no global state retained."""
    token = _enabled.set(True)
    try:
        with settlement_cache_scope():
            yield
    finally:
        _enabled.reset(token)


@contextlib.contextmanager
def settlement_cache_scope() -> collections.abc.Generator[None]:
    """Share a cache across nested analyses, but not subsequent requests."""
    if _cache.get() is not None:
        yield
        return
    token = _cache.set({})
    try:
        yield
    finally:
        _cache.reset(token)


def invalidate_settlement_cache() -> None:
    """Invalidate previously settled routes after new realisation failures."""
    cache = _cache.get()
    if cache is not None:
        cache.clear()


def memoized_settlement(
    assignments: collections.abc.Mapping[str, BiltModelConstituent],
    available: collections.abc.Set[str],
    backend: Backend,
    target: str,
    compute: collections.abc.Callable[[], SettlementResult],
) -> SettlementResult:
    """Memoize successful or unresolved settlements for an exact structural state."""
    cache = _cache.get()
    if not _enabled.get() or cache is None:
        return compute()
    pairs = tuple(sorted(assignments.items()))
    key: SettlementKey = (
        id(backend),
        target,
        frozenset(available),
        tuple((name, id(operation)) for name, operation in pairs),
    )
    if (entry := cache.get(key)) is not None:
        return entry[1]
    settled = compute()
    if len(cache) >= 256:  # noqa: PLR2004 - bounded per-analysis cache
        del cache[next(iter(cache))]
    # Retain identity references for the whole request so id() cannot be reused.
    cache[key] = (tuple(operation for _, operation in pairs), settled)
    return settled
