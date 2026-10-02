"""Check dimensionality across an implementation's interface and constituents by shared port name."""

__all__ = ["ResolvedDimensionality", "check_dimensional_consistency"]

import dataclasses
import typing

from bilt.core._equation import BasicEquation, check_homogeneity
from bilt.core._errors import BiltEquationDimensionalityError, BiltImplementationDimensionalityError
from bilt.core._port_declaration import declared_dimensionalities

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._interface import Interface
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.units import Dimension


@dataclasses.dataclass
class ResolvedDimensionality:
    """The dimensionality a port resolves to, and where it came from."""

    dimensionality: Dimension
    source: str


def check_dimensional_consistency(
    system: Interface,
    constituents: collections.abc.Iterable[BiltModelConstituent],
) -> dict[str, ResolvedDimensionality]:
    """Demand that each port of an implementation can take one dimensionality that every equation agrees with.

    Return the dimensionality each port resolves to and where it came from, leaving out ports that resolve to none.

    Ports are inferred one at a time, so a contradiction carried only by a combination of unresolved ports goes
    unreported: `x * y == length` and `x * y == mass` pass until something resolves `x` or `y`.
    """
    ordered = sorted(constituents, key=lambda constituent: constituent.__bilt_name__)
    resolution = _Resolution(system)
    for owner in (system, *ordered):
        for port, dimensionality in declared_dimensionalities(owner.__bilt_port_declarations__).items():
            resolution.declare(port, dimensionality, owner.__bilt_name__)

    equations = [constituent for constituent in ordered if isinstance(constituent, BasicEquation)]
    pending = dict.fromkeys(equations)
    while pending:
        equation = next(iter(pending))
        del pending[equation]
        for port in resolution.infer_from(equation):
            pending.update(dict.fromkeys(other for other in equations if port in other.__bilt_relation_ports__))

    return resolution.resolved


class _Resolution:
    """The dimensionality each port of an implementation resolves to, and where it came from."""

    def __init__(self, system: Interface) -> None:
        self._system = system
        self._dimensionalities: dict[str, Dimension] = {}
        self._sources: dict[str, str] = {}

    @property
    def resolved(self) -> dict[str, ResolvedDimensionality]:
        """The dimensionality of each port resolved so far, and where it came from."""
        return {
            port: ResolvedDimensionality(dimensionality, self._sources[port])
            for port, dimensionality in self._dimensionalities.items()
        }

    def declare(self, port: str, dimensionality: Dimension, owner_name: str) -> None:
        """Record a declared dimensionality, rejecting one that contradicts an earlier declaration."""
        source = f"declared by {owner_name}"
        known = self._dimensionalities.setdefault(port, dimensionality)
        self._sources.setdefault(port, source)
        if known is not dimensionality:
            msg = (
                f"gives port {port!r} conflicting dimensionalities:\n"
                f"  {known}: {self._sources[port]}\n  {dimensionality}: {source}"
            )
            raise BiltImplementationDimensionalityError(self._system.__bilt_name__, msg)

    def infer_from(self, equation: BasicEquation[object, ...]) -> list[str]:
        """Demand that an equation is homogeneous given its resolved ports; record and return the ports it implies."""
        known = {
            port: self._dimensionalities[port]
            for port in equation.__bilt_parameters__
            if port in self._dimensionalities
        }
        try:
            implied = check_homogeneity(equation, known)
        except BiltEquationDimensionalityError as error:
            given = "\n".join(
                f"  {port!r}: {dimensionality}: {self._sources[port]}" for port, dimensionality in sorted(known.items())
            )
            msg = f"makes equation {equation.__bilt_name__!r} dimensionally inconsistent, given\n{given}\n{error}"
            raise BiltImplementationDimensionalityError(self._system.__bilt_name__, msg) from error

        inputs = [repr(port) for port in sorted(equation.__bilt_relation_ports__) if port in known]
        source = f"inferred by {equation.__bilt_name__}"
        if inputs:
            source += f" from {', '.join(inputs)}"

        for port, dimensionality in implied.items():
            self._dimensionalities[port] = dimensionality
            self._sources[port] = source

        return list(implied)
