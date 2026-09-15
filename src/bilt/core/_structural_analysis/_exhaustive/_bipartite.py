__all__ = ["BipartiteGraph"]
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

    from bilt._backend._base_backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent

type Attribute = str
type Node = Attribute | PlanStep


class BipartiteGraph:
    def __init__(self, constituents: collections.abc.Iterable[BiltModelConstituent], backend: Backend) -> None:
        self.graph = nx.DiGraph()
        self._backend = backend
        for constituent in constituents:
            self._add_constituent(constituent)

    def required_inputs_for_step(self, plan_step: PlanStep) -> frozenset[str] | None:
        """Return the attributes a plan step consumes to produce its output.

        None means the step cannot produce its output at all: a sub-system has no internal route
        from its other ports to the one being asked for.
        """
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
                return sub_system.required_parent_inputs(plan_step.output, self._backend)
            case _:
                msg = f"Constituent type {type(plan_step.operation)} is not supported."
                raise NotImplementedError(msg)

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
        for param in relation.__bilt_parameters__:
            self.graph.add_node(param, bipartite="attribute")

        for param in relation.__bilt_parameters__:
            plan_step = PlanStep(operation=relation, output=param)
            self.graph.add_node(plan_step, bipartite="constituent")
            self.graph.add_edge(plan_step, param)
            for param2 in relation.__bilt_parameters__:
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
        """Remove a step that failed realization, unless its sub-system still names another way."""
        match plan_step.operation:
            case SubSystemConstituent() if self.required_inputs_for_step(plan_step) is not None:
                return
            case _:
                self.graph.remove_node(plan_step)

    def search(
        self,
        available: set[str],
        output: Node,
        visited: set[Node] | None = None,
    ) -> list[PlanStep] | None:
        """Depth-first search for a path from available inputs to the desired output."""
        if output not in self.graph:
            msg = f"Output {output} is not in the graph. Please check the constituents provided."
            raise BiltSystemAnalysisError(msg)
        if output in available:
            return []

        on_path: set[Node] = set() if visited is None else visited
        if output in on_path:
            return None

        on_path.add(output)
        try:
            if self.graph.nodes[output]["bipartite"] == "constituent":
                return self._search_constituent(typing.cast("PlanStep", output), available, on_path)
            return self._search_attribute(typing.cast("Attribute", output), available, on_path)
        finally:
            on_path.discard(output)

    def _search_constituent(
        self,
        plan_step: PlanStep,
        available: set[str],
        visited: set[Node],
    ) -> list[PlanStep] | None:
        """Search for a path through a constituent node, resolving every input it needs.

        Inputs are visited in sorted order so a given query always yields the same plan ordering.
        """
        required = self.required_inputs_for_step(plan_step)
        if required is None:
            return None

        path: list[PlanStep] = []
        included: set[PlanStep] = set()
        for neighbor in sorted(required):
            sub_path = self.search(available, neighbor, visited)
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
    ) -> list[PlanStep] | None:
        """Search for a path through an attribute node, trying each constituent that produces it."""
        for neighbor in self.graph.predecessors(attribute):
            sub_path = self.search(available, neighbor, visited)
            if sub_path is not None:
                return sub_path
        return None
