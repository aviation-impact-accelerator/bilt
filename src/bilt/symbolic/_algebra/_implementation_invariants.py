"""Experimental automatic invariant discovery from a Bilt implementation.

This inspects direct primitive equations only. Nested-system equations,
conditional branches and transforms are not included.
"""

__all__ = ["discover_implementation_invariants"]

import typing

from bilt.core._equation import BasicEquation
from bilt.symbolic._algebra._invariants import DiscoveredInvariant, discover_boundary_invariants

if typing.TYPE_CHECKING:
    from bilt.units.dimensions._dimensions import Dimension
    from bilt.core._system._implementation import Implementation


def discover_implementation_invariants(
    implementation: Implementation,
    *,
    dimension: Dimension | None = None,
    max_degree: int = 4,
    max_terms: int = 256,
) -> tuple[DiscoveredInvariant, ...]:
    """Discover additive identities over declared interface fields.

    The system interface supplies boundary names automatically. When a
    dimensionality is supplied, only boundary ports of that physical dimension
    may appear in discovered identities. Plain float fields cannot be
    interpreted as dimensional physical quantities.

    Only direct primitive equations are analysed. This deliberately does not
    include child implementation equations or transforms.
    """
    interface = implementation.system
    declarations = interface.__bilt_port_declarations__
    boundary = frozenset(
        name
        for name in interface.__bilt_parameters__
        if dimension is None or (name in declarations and declarations[name].dimensionality == dimension)
    )
    equations = tuple(
        (component.__bilt_name__, component.__bilt_equation_tree__)
        for component in implementation._components.values()  # noqa: SLF001
        if isinstance(component, BasicEquation)
    )
    return discover_boundary_invariants(
        equations,
        boundary,
        max_degree=max_degree,
        max_terms=max_terms,
    )
