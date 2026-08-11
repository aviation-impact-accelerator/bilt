__all__ = ["BipartiteGraph", "required_inputs_for_target"]
import typing
from typing import TYPE_CHECKING

import networkx as nx

from bilt.core._equation import BasicEquation
from bilt.core._structural_analysis._errors import BiltSystemAnalysisError
from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import BasicTransform

if TYPE_CHECKING:
    import collections.abc

    from bilt.core._model_constituent import BiltModelConstituent

type Attribute = str
type Node = Attribute | PlanStep


def required_inputs_for_target(plan_step: PlanStep) -> frozenset[str] | None:
    """Return the inputs a plan step consumes, or None when no path exists to compute it."""
    match plan_step.operation:
        case BasicTransform() as transform:
            if plan_step.output != transform.__bilt_name__:
                msg = f"{transform!r} does not produce {plan_step.output!r}."
                raise BiltSystemAnalysisError(msg)
            return frozenset(transform.__bilt_parameters__)
        case BasicEquation() as equation:
            if plan_step.output not in equation.__bilt_parameters__:
                msg = f"{equation!r} does not have parameter {plan_step.output!r}."
                raise BiltSystemAnalysisError(msg)
            return frozenset(equation.__bilt_parameters__) - {plan_step.output}
        case SubSystemConstituent() as sub_system:
            child_output = sub_system.unqualify(plan_step.output)
            if child_output is None:
                msg = f"{sub_system!r} does not have port {plan_step.output!r}."
                raise BiltSystemAnalysisError(msg)
            available_child_ports = set(sub_system.interface.__bilt_parameters__) - {child_output}
            required_child_ports = sub_system.implementation.analyser.required_inputs_for_target(
                available_child_ports, child_output
            )
            if required_child_ports is None:
                return None
            return frozenset(sub_system.qualify(port) for port in required_child_ports)
        case _:
            msg = f"Constituent type {type(plan_step.operation)} is not supported."
            raise NotImplementedError(msg)


class BipartiteGraph:
    def __init__(self, constituents: collections.abc.Iterable[BiltModelConstituent]) -> None:
        self.graph = nx.DiGraph()
        for constituent in constituents:
            self._add_constituent(constituent)

    def _add_constituent(self, constituent: BiltModelConstituent) -> None:
        """Add a new constituent to the graph."""
        match constituent:
            case BasicTransform() as transform:
                self._add_transform(transform)
            case BasicEquation() | SubSystemConstituent() as relation:
                self._add_relation(relation)
            case _:
                msg = f"Constituent type {type(constituent)} is not supported in the bipartite graph."
                raise NotImplementedError(msg)

    def _add_relation(self, relation: BasicEquation | SubSystemConstituent) -> None:
        """Add a constituent that can be solved for any one of its parameters given the others."""
        parameters = relation.__bilt_parameters__
        self.graph.add_nodes_from(parameters, bipartite="attribute")

        for output in parameters:
            plan_step = PlanStep(operation=relation, output=output)
            self.graph.add_node(plan_step, bipartite="constituent")
            self.graph.add_edge(plan_step, output)
            self.graph.add_edges_from((other, plan_step) for other in parameters if other != output)

    def _add_transform(self, transform: BasicTransform) -> None:
        """Add a transform, which produces one fixed output from all of its parameters."""
        plan_step = PlanStep(operation=transform, output=transform.__bilt_name__)
        self.graph.add_node(plan_step, bipartite="constituent")
        self.graph.add_nodes_from((transform.__bilt_name__, *transform.__bilt_parameters__), bipartite="attribute")
        self.graph.add_edge(plan_step, transform.__bilt_name__)
        self.graph.add_edges_from((param, plan_step) for param in transform.__bilt_parameters__)

    def search(
        self,
        available: set[str],
        output: Node,
        visited: set[Node] | None = None,
        excluded: frozenset[PlanStep] = frozenset(),
    ) -> list[PlanStep] | None:
        """Depth-first search for a path from available inputs to the desired output.

        `excluded` names plan steps to route around. It is per-query on purpose: a step that
        cannot be realized for one set of inputs and one backend may be fine for another, so
        the graph itself is never modified.
        """
        if output not in self.graph:
            msg = f"Output {output} is not in the graph. Please check the constituents provided."
            raise BiltSystemAnalysisError(msg)
        if output in available:
            return []
        if output in excluded:
            return None

        on_path = set() if visited is None else visited
        if output in on_path:
            return None

        on_path.add(output)
        try:
            if self.graph.nodes[output]["bipartite"] == "constituent":
                return self._search_constituent(typing.cast("PlanStep", output), available, on_path, excluded)
            return self._search_attribute(typing.cast("Attribute", output), available, on_path, excluded)
        finally:
            on_path.discard(output)

    def _search_constituent(
        self, plan_step: PlanStep, available: set[str], visited: set[Node], excluded: frozenset[PlanStep]
    ) -> list[PlanStep] | None:
        """Search for a path through a constituent node, resolving every input it needs."""
        required = required_inputs_for_target(plan_step)
        if required is None:
            return None

        path: list[PlanStep] = []
        included: set[PlanStep] = set()
        for neighbor in required:
            sub_path = self.search(available, neighbor, visited, excluded)
            if sub_path is None:
                return None
            for step in sub_path:
                if step not in included:
                    included.add(step)
                    path.append(step)
        path.append(plan_step)
        return path

    def _search_attribute(
        self, attribute: Attribute, available: set[str], visited: set[Node], excluded: frozenset[PlanStep]
    ) -> list[PlanStep] | None:
        """Search for a path through an attribute node."""
        for neighbor in self.graph.predecessors(attribute):
            sub_path = self.search(available, neighbor, visited, excluded)
            if sub_path is not None:
                return sub_path
        return None
