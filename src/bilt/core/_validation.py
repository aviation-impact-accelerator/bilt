"""Collect independent definition errors when running the model checker."""

__all__ = [
    "Diagnostic",
    "code_for",
    "collecting_validation_errors",
    "validate_independently",
]

import contextlib
import contextvars
import dataclasses
import typing

from bilt._base_error import BiltError
from bilt.core._errors import (
    BiltCollectedValidationError,
    BiltEquationDimensionalityError,
    BiltImplementationDimensionalityError,
    BiltPortDeclarationError,
)

if typing.TYPE_CHECKING:
    import collections.abc


@dataclasses.dataclass(frozen=True, slots=True)
class Diagnostic:
    """One reported definition error with its source location and stable category code."""

    location: str
    message: str
    code: str


def code_for(error: Exception) -> str:
    """Assign stable codes to the first supported validation categories."""
    if isinstance(error, BiltEquationDimensionalityError):
        return "BILT001"
    if isinstance(error, BiltImplementationDimensionalityError):
        return "BILT002"
    if isinstance(error, BiltPortDeclarationError):
        return "BILT003"
    return "BILT999"


_collecting = contextvars.ContextVar("bilt_collect_validation_errors", default=False)


@contextlib.contextmanager
def collecting_validation_errors() -> collections.abc.Generator[None]:
    """Collect independent errors during checks, without changing normal model construction."""
    token = _collecting.set(True)
    try:
        yield
    finally:
        _collecting.reset(token)


def validate_independently(checks: collections.abc.Iterable[collections.abc.Callable[[], object]]) -> None:
    """Run independent checks; preserve first-error behaviour outside the checker.

    Callers must stop model construction when this raises a group: an invalid
    equation or implementation must not be registered or evaluated.
    """
    if not _collecting.get():
        for check in checks:
            check()
        return

    errors: list[BiltError] = []
    for check in checks:
        try:
            check()
        except BiltError as error:
            errors.append(error)
    if errors:
        raise BiltCollectedValidationError(errors)
