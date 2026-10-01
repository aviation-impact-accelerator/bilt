"""Unify dimensionality across an implementation's interface and constituents by shared port name."""

__all__ = ["unify_dimensionalities"]

import typing

from bilt.core._dimensional_analysis import dimensionality_of
from bilt.core._equation import BasicEquation, check_homogeneity
from bilt.core._errors import BiltEquationDimensionalityError, BiltImplementationDimensionalityError
from bilt.symbolic._isolation import isolate
from bilt.symbolic._isolation._errors import BiltIsolationError
from bilt.symbolic._walk import subtrees
from bilt.units._errors import BiltDimensionalityError

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._interface import Interface
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.symbolic._tree import Tree
    from bilt.units import Dimension

type _Finding = tuple[str, Dimension, str]


def unify_dimensionalities(
    system: Interface,
    constituents: collections.abc.Iterable[BiltModelConstituent],
) -> dict[str, Dimension]:
    """Return the dimensionality each port of an implementation resolves to, leaving out ports nothing constrains.

    Raises:
        BiltImplementationDimensionalityError: Two sources give a port different dimensionalities, or a
            constituent is dimensionally inconsistent given the dimensionalities of its ports.
    """
    ordered = sorted(constituents, key=lambda constituent: constituent.__bilt_name__)
    resolved: dict[str, Dimension] = {}
    sources: dict[str, str] = {}
    declared = (
        (port, declaration.dimensionality, f"declared by {owner.__bilt_name__}")
        for owner in (system, *ordered)
        for port, declaration in owner.__bilt_port_declarations__.items()
        if declaration.dimensionality is not None
    )
    _absorb(system, resolved, sources, declared)

    solved = {
        constituent: _solutions(constituent, resolved)
        for constituent in ordered
        if isinstance(constituent, BasicEquation)
    }
    while True:
        count = len(resolved)
        _absorb(
            system,
            resolved,
            sources,
            (finding for equation, solutions in solved.items() for finding in _inferred(equation, solutions, resolved)),
        )
        for equation in solved:
            _check_consistency(system, equation, resolved, sources)
        if len(resolved) == count:
            return resolved


def _solutions(equation: BasicEquation[object, ...], declared: collections.abc.Container[str]) -> dict[str, Tree]:
    """Isolate each undeclared port an equation can be solved for."""
    solutions: dict[str, Tree] = {}
    for port, symbol in equation.__bilt_equation_symbols__.items():
        if port in declared:
            continue
        try:
            solutions[port] = isolate(equation.__bilt_equation_tree__, symbol).rhs.rep
        except BiltIsolationError:
            continue
    return solutions


def _inferred(
    equation: BasicEquation[object, ...],
    solutions: collections.abc.Mapping[str, Tree],
    resolved: collections.abc.Mapping[str, Dimension],
) -> collections.abc.Iterator[_Finding]:
    """Work out each dimensionality the resolved ports newly determine for one of an equation's ports."""
    for port, rhs in solutions.items():
        try:
            dimensionality = dimensionality_of(rhs, resolved)
        except BiltDimensionalityError:
            continue
        if dimensionality is None or dimensionality is resolved.get(port):
            continue
        occurring = subtrees(rhs)
        operands = sorted(
            other
            for other, symbol in equation.__bilt_equation_symbols__.items()
            if other in resolved and symbol.rep in occurring
        )
        yield port, dimensionality, f"inferred by {equation.__bilt_name__} from {', '.join(map(repr, operands))}"


def _absorb(
    system: Interface,
    resolved: dict[str, Dimension],
    sources: dict[str, str],
    findings: collections.abc.Iterable[_Finding],
) -> None:
    """Record each finding's dimensionality and source against a port that has none yet.

    Raises:
        BiltImplementationDimensionalityError: A finding gives a port a different dimensionality from the one it has.
    """
    for port, dimensionality, source in findings:
        known = resolved.setdefault(port, dimensionality)
        sources.setdefault(port, source)
        if known is not dimensionality:
            msg = f"gives port {port!r} conflicting dimensionalities:\n  {known}: {sources[port]}\n  {dimensionality}: {source}"
            raise BiltImplementationDimensionalityError(system.__bilt_name__, msg)


def _check_consistency(
    system: Interface,
    equation: BasicEquation[object, ...],
    resolved: collections.abc.Mapping[str, Dimension],
    sources: collections.abc.Mapping[str, str],
) -> None:
    """Demand that an equation's relation is dimensionally homogeneous given its resolved ports.

    Raises:
        BiltImplementationDimensionalityError: It is not, citing where its resolved ports' dimensionalities came from.
    """
    known = {port: resolved[port] for port in equation.__bilt_parameters__ if port in resolved}
    try:
        check_homogeneity(equation, known)
    except BiltEquationDimensionalityError as error:
        given = "\n".join(
            f"  {port!r}: {dimensionality}: {sources[port]}" for port, dimensionality in sorted(known.items())
        )
        msg = f"makes equation {equation.__bilt_name__!r} dimensionally inconsistent, given\n{given}\n{error}"
        raise BiltImplementationDimensionalityError(system.__bilt_name__, msg) from error
