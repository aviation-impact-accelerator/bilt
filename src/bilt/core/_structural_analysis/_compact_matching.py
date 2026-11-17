"""Match regular tensor dependency patterns without expanding coordinate edges.

The exact fast paths are deliberately narrow: one residual family, one
unknown family and a provable pointwise or complete bipartite relationship.
Everything else uses the existing Dulmage-Mendelsohn analysis on a bounded
coordinate graph. Neither outcome proves numerical invertibility.
"""

__all__ = ["CompactStructuralMatch", "match_compact_incidence"]

import dataclasses
import math
import typing

from bilt.core._structural_analysis._causalization_block_solver._dm import DMPartition, dulmage_mendelsohn

if typing.TYPE_CHECKING:
    from bilt.core._domain_registry import DomainRegistry
    from bilt.core._structural_analysis._compact_incidence import CompactIncidence, DependencyPath
    from bilt.symbolic._dimensions import Dim


type StructuralStatus = typing.Literal["well-determined", "under-determined", "over-determined", "mixed"]
type MatchingPattern = typing.Literal["pointwise", "complete", "expanded"]


@dataclasses.dataclass(frozen=True, slots=True)
class CompactStructuralMatch:
    """Structural matching result; no numerical or execution plan is implied.

    A pointwise system contains one independent equation for every tensor
    coordinate. A complete square system forms one potentially coupled block.
    General small patterns retain their exact coordinate-level DM partition.
    """

    pattern: MatchingPattern
    status: StructuralStatus
    residual_coordinates: int
    unknown_coordinates: int
    independent_blocks: int | None
    partition: DMPartition | None = None

    @property
    def well_determined(self) -> bool:
        return self.status == "well-determined"


def match_compact_incidence(
    incidence: CompactIncidence,
    bindings: DomainRegistry,
    *,
    max_nodes: int = 4096,
    max_edges: int = 100_000,
) -> CompactStructuralMatch:
    """Match regular patterns symbolically, falling back to bounded DM matching.

    This is not yet used by production execution planning. It provides the
    matching stage for the compact representation without changing Bilt's
    existing fast path for scalar equations or subsystem boundaries.
    """
    residual_count = sum(_size(row.domain.dims, bindings) for row in incidence.residuals)
    unknown_count = sum(_size(domain.dims, bindings) for _, domain in incidence.unknown_domains)

    regular = _regular_pattern(incidence, bindings)
    if regular == "pointwise":
        return CompactStructuralMatch(
            pattern="pointwise",
            status="well-determined",
            residual_coordinates=residual_count,
            unknown_coordinates=unknown_count,
            independent_blocks=residual_count,
        )
    if regular == "complete":
        status: StructuralStatus
        if residual_count == unknown_count:
            status = "well-determined"
        elif residual_count < unknown_count:
            status = "under-determined"
        else:
            status = "over-determined"
        return CompactStructuralMatch(
            pattern="complete",
            status=status,
            residual_coordinates=residual_count,
            unknown_coordinates=unknown_count,
            independent_blocks=1 if status == "well-determined" else None,
        )

    graph, unknown_nodes = incidence.materialize(bindings, max_nodes=max_nodes, max_edges=max_edges)
    partition = dulmage_mendelsohn(graph, unknown_nodes)
    return CompactStructuralMatch(
        pattern="expanded",
        status=_partition_status(partition),
        residual_coordinates=residual_count,
        unknown_coordinates=unknown_count,
        independent_blocks=None,
        partition=partition,
    )


def _regular_pattern(incidence: CompactIncidence, bindings: DomainRegistry) -> MatchingPattern | None:
    if len(incidence.unknown_domains) != 1 or len(incidence.residuals) != 1:
        return None
    name, unknown_domain = incidence.unknown_domains[0]
    residual = incidence.residuals[0]
    if not residual.paths or any(path.unknown != name for path in residual.paths):
        return None

    # An empty residual or unknown dimension cannot form a nonempty matching.
    if not _size(residual.domain.dims, bindings) or not _size(unknown_domain.dims, bindings):
        return None

    kinds = tuple(_classify_path(path, residual.domain.dims, unknown_domain.dims, bindings) for path in residual.paths)
    if "complete" in kinds:
        return "complete"
    if kinds and all(kind == "pointwise" for kind in kinds):
        return "pointwise"
    return None


def _classify_path(
    path: DependencyPath,
    residual_dims: tuple[Dim, ...],
    unknown_dims: tuple[Dim, ...],
    bindings: DomainRegistry,
) -> MatchingPattern | None:
    """Track which unknown axes are fixed by a particular residual coordinate.

    A dimension inherited from the residual remains pointwise. A dimension
    introduced by a reduction is free to range over its entire bound extent.
    Selection, missing axes and unrecognised relationships are never guessed
    to be regular.
    """
    origins: dict[Dim, Dim | None] = {dim: dim for dim in residual_dims}
    for relation in path.relations:
        if relation.selector is not None or set(relation.output_domain.dims) != set(origins):
            return None
        next_origins: dict[Dim, Dim | None] = {}
        for dim in relation.operand_domain.dims:
            if dim in origins:
                next_origins[dim] = origins[dim]
            elif dim in relation.reduced_axes:
                if bindings.size(dim) == 0:
                    return None
                next_origins[dim] = None
            else:
                return None
        origins = next_origins

    if set(origins) == set(unknown_dims):
        if set(unknown_dims) == set(residual_dims) and all(origins[dim] == dim for dim in unknown_dims):
            return "pointwise"
        if all(origins[dim] is None for dim in unknown_dims):
            return "complete"
    return None


def _size(dims: tuple[Dim, ...], bindings: DomainRegistry) -> int:
    return math.prod(bindings.size(dim) for dim in dims)


def _partition_status(partition: DMPartition) -> StructuralStatus:
    if partition.underdetermined and partition.overdetermined:
        return "mixed"
    if partition.underdetermined:
        return "under-determined"
    if partition.overdetermined:
        return "over-determined"
    return "well-determined"
