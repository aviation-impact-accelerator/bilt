"""Implementation-owned numerical settings, separate from model boundary values."""

import contextlib
import contextvars
import dataclasses
import types
import typing

from bilt._backend._nonlinear_solve import NonlinearControls
from bilt.core._equation import BasicEquation

if typing.TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from bilt._backend._nonlinear_solve import NonlinearSolver


@dataclasses.dataclass(frozen=True)
class VariableOptions:
    """An initial value and positive physical scale; neither fixes a model input."""

    initial: object
    scale: object


@dataclasses.dataclass(frozen=True, eq=False)
class RootOptions:
    """Settings for local root solves, addressed by ports and original equations."""

    variables: Mapping[str, VariableOptions]
    residual_scales: Mapping[BasicEquation, object]
    solver: NonlinearSolver | None = None
    controls: NonlinearControls = dataclasses.field(default_factory=NonlinearControls)

    def __post_init__(self) -> None:
        if any(not isinstance(name, str) or not name for name in self.variables):
            message = "Root variable settings require nonempty port names."
            raise ValueError(message)
        if any(not isinstance(value, VariableOptions) for value in self.variables.values()):
            message = "Root variables must contain VariableOptions values."
            raise TypeError(message)
        if any(not isinstance(source, BasicEquation) for source in self.residual_scales):
            message = "Residual scales must be keyed by original equation objects."
            raise TypeError(message)
        object.__setattr__(self, "variables", types.MappingProxyType(dict(self.variables)))
        object.__setattr__(self, "residual_scales", types.MappingProxyType(dict(self.residual_scales)))


@dataclasses.dataclass(frozen=True, eq=False)
class EvaluationOptions:
    """Immutable numerical policy attached to an implementation view.

    Mapping contents are copied by RootOptions. Numerical values and solver
    settings are runtime data, not extra boundary constraints or compiled code.
    """

    nonlinear: RootOptions | None = None


_options: contextvars.ContextVar[EvaluationOptions | None] = contextvars.ContextVar(
    "bilt_evaluation_options", default=None
)


def current_options() -> EvaluationOptions | None:
    return _options.get()


def current_root_options() -> RootOptions | None:
    options = current_options()
    return None if options is None else options.nonlinear


def numerical_signature() -> tuple[frozenset[str], frozenset[BasicEquation]] | None:
    """Only settings availability affects local preparation; values remain runtime data."""
    roots = current_root_options()
    return None if roots is None else (frozenset(roots.variables), frozenset(roots.residual_scales))


@contextlib.contextmanager
def _using_options(options: EvaluationOptions | None) -> Generator[None]:
    """Install an implementation's options for internal planning and execution only."""
    token = _options.set(options)
    try:
        yield
    finally:
        _options.reset(token)
