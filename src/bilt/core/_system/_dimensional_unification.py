"""Unify dimensionality across an implementation's interface and constituents by shared port name."""

__all__ = ["unify_dimensionalities"]

import typing

from bilt.core._dimensional_analysis import implied_dimensionality
from bilt.core._equation import BasicEquation, check_homogeneity
from bilt.core._errors import BiltEquationDimensionalityError, BiltImplementationDimensionalityError
from bilt.units._errors import BiltDimensionalityError

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._interface import Interface
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.units import Dimension

type _Declaration = tuple[str, Dimension, str]


def unify_dimensionalities(
    system: Interface,
    constituents: collections.abc.Iterable[BiltModelConstituent],
) -> dict[str, Dimension]:
    """Return the dimensionality each port of an implementation resolves to, leaving out ports nothing constrains."""
    ordered = sorted(constituents, key=lambda constituent: constituent.__bilt_name__)
    resolved: dict[str, Dimension] = {}
    sources: dict[str, str] = {}
    declarations = (
        (port, declaration.dimensionality, f"declared by {owner.__bilt_name__}")
        for owner in (system, *ordered)
        for port, declaration in owner.__bilt_port_declarations__.items()
        if declaration.dimensionality is not None
    )
    _declare(system, resolved, sources, declarations)

    equations = [constituent for constituent in ordered if isinstance(constituent, BasicEquation)]
    asked = dict.fromkeys(equations, -1)
    checked = dict.fromkeys(equations, -1)
    while True:
        count = len(resolved)
        for equation in equations:
            if _more_resolved(equation, resolved, asked):
                for port in equation.__bilt_relation_ports__:
                    if port not in resolved:
                        _infer(equation, port, resolved, sources)
        for equation in equations:
            if _more_resolved(equation, resolved, checked):
                _check_consistency(system, equation, resolved, sources)
        if len(resolved) == count:
            return resolved


def _more_resolved(
    equation: BasicEquation[object, ...],
    resolved: collections.abc.Container[str],
    counts: dict[BasicEquation[object, ...], int],
) -> bool:
    """Return whether more of an equation's ports are resolved than `counts` records, and record how many are."""
    count = sum(port in resolved for port in equation.__bilt_relation_ports__)
    if counts[equation] == count:
        return False
    counts[equation] = count
    return True


def _infer(
    equation: BasicEquation[object, ...],
    port: str,
    resolved: dict[str, Dimension],
    sources: dict[str, str],
) -> None:
    """Record the dimensionality an equation implies for an unresolved port, and the resolved ports it came from."""
    try:
        dimensionality = implied_dimensionality(equation.__bilt_equation_tree__, port, resolved)
    except BiltDimensionalityError:
        return
    if dimensionality is None:
        return
    inputs = [repr(other) for other in sorted(equation.__bilt_relation_ports__) if other in resolved]
    resolved[port] = dimensionality
    sources[port] = f"inferred by {equation.__bilt_name__}"
    if inputs:
        sources[port] += f" from {', '.join(inputs)}"


def _declare(
    system: Interface,
    resolved: dict[str, Dimension],
    sources: dict[str, str],
    declarations: collections.abc.Iterable[_Declaration],
) -> None:
    """Record each declared dimensionality and its source, rejecting one that contradicts an earlier declaration."""
    for port, dimensionality, source in declarations:
        known = resolved.setdefault(port, dimensionality)
        sources.setdefault(port, source)
        if known is not dimensionality:
            msg = (
                f"gives port {port!r} conflicting dimensionalities:\n"
                f"  {known}: {sources[port]}\n  {dimensionality}: {source}"
            )
            raise BiltImplementationDimensionalityError(system.__bilt_name__, msg)


def _check_consistency(
    system: Interface,
    equation: BasicEquation[object, ...],
    resolved: collections.abc.Mapping[str, Dimension],
    sources: collections.abc.Mapping[str, str],
) -> None:
    """Demand that an equation's relation is dimensionally homogeneous given its resolved ports."""
    known = {port: resolved[port] for port in equation.__bilt_parameters__ if port in resolved}
    try:
        check_homogeneity(equation, known)
    except BiltEquationDimensionalityError as error:
        given = "\n".join(
            f"  {port!r}: {dimensionality}: {sources[port]}" for port, dimensionality in sorted(known.items())
        )
        msg = f"makes equation {equation.__bilt_name__!r} dimensionally inconsistent, given\n{given}\n{error}"
        raise BiltImplementationDimensionalityError(system.__bilt_name__, msg) from error
