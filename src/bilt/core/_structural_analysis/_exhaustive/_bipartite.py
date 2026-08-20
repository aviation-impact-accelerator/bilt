__all__ = ["BipartiteGraph", "required_inputs_for_target"]
import typing
from typing import TYPE_CHECKING

import networkx as nx

from bilt.core._equation import BasicEquation
from bilt.core._structural_analysis._errors import BiltSystemAnalysisError
from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._transform import BasicTransform

if TYPE_CHECKING:
    import collections.abc

    from bilt.core._model_constituent import BiltModelConstituent

type Attribute = str
type Node = Attribute | PlanStep


def required_inputs_for_target(plan_step: PlanStep) -> frozenset[str]:
    """Return the attributes a plan step consumes to produce its output."""
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
            case BasicEquation() as equation:
                self._add_equation(equation)
            case _:
                msg = f"Constituent type {type(constituent)} is not supported in the bipartite graph."
                raise NotImplementedError(msg)

    def _add_equation(self, equation: BasicEquation) -> None:
        """Add a new equation to the graph."""
        for param in equation.__bilt_parameters__:
            self.graph.add_node(param, bipartite="attribute")

        for param in equation.__bilt_parameters__:
            plan_step = PlanStep(operation=equation, output=param)
            self.graph.add_node(plan_step, bipartite="constituent")
            self.graph.add_edge(plan_step, param)
            for param2 in equation.__bilt_parameters__:
                if param2 != param:
                    self.graph.add_edge(param2, plan_step)

    def _add_transform(self, transform: BasicTransform) -> None:
        """Add a new transform to the graph."""
        plan_step = PlanStep(operation=transform, output=transform.__bilt_name__)
        self.graph.add_node(plan_step, bipartite="constituent")
        self.graph.add_node(transform.__bilt_name__, bipartite="attribute")
        self.graph.add_edge(plan_step, transform.__bilt_name__)
        for param in transform.__bilt_parameters__:
            self.graph.add_node(param, bipartite="attribute")
            self.graph.add_edge(param, plan_step)

    def remove_failed_plan_step(
        self,
        plan_step: PlanStep,
    ) -> None:
        """If an System or Equation is found to be unable to be used in a certain direction after a realization, remove it from the graph."""
        self.graph.remove_node(plan_step)

    def search(
        self,
        available: set[str],
        output: Node,
        visited: set[Node] | None = None,
        excluded: frozenset[PlanStep] = frozenset(),
    ) -> list[PlanStep] | None:
        """Depth-first search for a path from available inputs to the desired output.

        ``excluded`` names plan steps to route around for this query only. Steps that can never
        be realized are removed from the graph outright; ``excluded`` is for failures that depend
        on the requested inputs or the concrete backend, which must not outlive the query.
        """
        if output not in self.graph:
            msg = f"Output {output} is not in the graph. Please check the constituents provided."
            raise BiltSystemAnalysisError(msg)
        if output in available:
            return []
        if output in excluded:
            return None

        on_path: set[Node] = set() if visited is None else visited
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
        self,
        plan_step: PlanStep,
        available: set[str],
        visited: set[Node],
        excluded: frozenset[PlanStep] = frozenset(),
    ) -> list[PlanStep] | None:
        """Search for a path through a constituent node, resolving every input it needs.

        Inputs are visited in sorted order so a given query always yields the same plan ordering.
        """
        path: list[PlanStep] = []
        included: set[PlanStep] = set()
        for neighbor in sorted(required_inputs_for_target(plan_step)):
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
        self,
        attribute: Attribute,
        available: set[str],
        visited: set[Node],
        excluded: frozenset[PlanStep] = frozenset(),
    ) -> list[PlanStep] | None:
        """Search for a path through an attribute node, trying each constituent that produces it."""
        for neighbor in self.graph.predecessors(attribute):
            sub_path = self.search(available, neighbor, visited, excluded)
            if sub_path is not None:
                return sub_path
        return None
