import typing

import rustworkx as rx

from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltOverconstrainedError,
    BiltUnderconstrainedError,
)
from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import BasicTransform

from ._dm import DMPartition, dulmage_mendelsohn

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend._base_backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import (
        FullIncidenceGraph,
    )


def validate_structure(
    graph: rx.PyGraph,
    attribute_nodes: frozenset[int],
    input_nodes: frozenset[int],
    *,
    required_nodes: frozenset[int] | None = None,
) -> DMPartition:
    """Validate a scoped constraint graph after fixing supplied inputs.

    The caller establishes the constraint scope. By default every unknown
    attribute is required. With required_nodes, deficiencies unrelated to
    those requested attributes are retained in the result without raising.
    """
    unknown_nodes = attribute_nodes - input_nodes

    reduced = graph.copy()
    reduced.remove_nodes_from(list(attribute_nodes & input_nodes))

    partition = dulmage_mendelsohn(reduced, unknown_nodes)

    if partition.overdetermined:
        constituents = sorted(str(reduced[node]) for node in partition.overdetermined - unknown_nodes)
        message = (
            f"Overconstrained problem involving constituents: "
            f"{', '.join(constituents)}. Reassess the supplied constituents."
        )
        raise BiltOverconstrainedError(message)

    required = unknown_nodes if required_nodes is None else required_nodes - input_nodes
    if partition.underdetermined & required:
        attributes = sorted(str(reduced[node]) for node in partition.underdetermined & unknown_nodes)
        message = f"Underconstrained problem involving attributes: {', '.join(attributes)}."
        raise BiltUnderconstrainedError(message)

    return partition


def validate_request_structure(
    incidence: FullIncidenceGraph,
    available: collections.abc.Set[str],
    output: str,
    backend: Backend,
) -> None:
    """Validate the requested constraint scope, stopping at supplied inputs.

    All relevant supplied equations remain constraints, irrespective of
    isolation or backend support. Transform input dependencies are preserved,
    but transforms enter the scope only through their declared output.
    Scopes crossing an opaque subsystem boundary are deferred to the existing
    route planner. Boundary routes are alternatives, not independent constraints.
    A child analyser validates its own scope when that route is evaluated.
    """
    if output in available:
        return

    producers: dict[str, list[BiltModelConstituent]] = {}
    for node in incidence.constituent_nodes:
        constituent = incidence.graph[node]
        outputs = (
            (constituent.__bilt_name__,) if isinstance(constituent, BasicTransform) else constituent.__bilt_parameters__
        )
        for attribute in outputs:
            producers.setdefault(attribute, []).append(constituent)

    offered = set(available) | set(producers)
    rows: dict[object, tuple[object, frozenset[str]]] = {}
    visited: set[str] = set()
    pending = [output]

    while pending:
        attribute = pending.pop()
        if attribute in available or attribute in visited:
            continue
        visited.add(attribute)

        for constituent in producers.get(attribute, ()):
            if isinstance(constituent, SubSystemConstituent):
                # Do not count alternative child routes as simultaneous equations.
                return
            key, payload, attributes = typing.cast(
                "tuple[object, object, frozenset[str]]",
                _constraint_row(constituent, attribute, offered, backend),
            )
            rows[key] = payload, attributes
            pending.extend(attributes - available)

    graph = rx.PyGraph(multigraph=False)
    attribute_nodes = {attribute: graph.add_node(attribute) for attribute in sorted(visited)}
    for payload, attributes in rows.values():
        node = graph.add_node(payload)
        for attribute in sorted(attributes - available):
            graph.add_edge(node, attribute_nodes[attribute], None)

    validate_structure(
        graph,
        frozenset(attribute_nodes.values()),
        frozenset(),
        required_nodes=frozenset({attribute_nodes[output]}),
    )


def _constraint_row(
    constituent: BiltModelConstituent,
    output: str,
    offered: set[str],
    backend: Backend,
) -> tuple[object, object, frozenset[str]] | None:
    if isinstance(constituent, SubSystemConstituent):
        child_inputs = constituent.required_inputs_for(
            output,
            backend,
            within=set(constituent.__bilt_parameters__) & offered,
        )
        if child_inputs is None:
            return None
        step = PlanStep(operation=constituent, output=output, child_inputs=child_inputs)
        return (id(constituent), output), step, frozenset({output, *step.consumed})

    attributes = frozenset(constituent.__bilt_parameters__)
    if isinstance(constituent, BasicTransform):
        attributes |= {constituent.__bilt_name__}
    return id(constituent), constituent, attributes
