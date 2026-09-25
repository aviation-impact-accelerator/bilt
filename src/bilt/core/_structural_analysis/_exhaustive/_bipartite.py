__all__ = ["BipartiteGraph"]

import dataclasses
import itertools
import typing
from typing import TYPE_CHECKING

import networkx as nx

from bilt.core._equation import BasicEquation
from bilt.core._errors import BiltAnalysisError
from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import BasicTransform

if TYPE_CHECKING:
    import collections.abc

    from bilt._backend._base_backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent

type Attribute = str
type Node = Attribute | PlanStep


def _flatten_sub_paths(sub_paths: collections.abc.Iterable[list[PlanStep]]) -> list[PlanStep]:
    """Flatten the paths that produced a step's inputs, keeping the first step to produce each attribute."""
    produced: dict[Attribute, PlanStep] = {}
    for step in itertools.chain.from_iterable(sub_paths):
        produced.setdefault(step.output, step)
    return list(produced.values())


class BipartiteGraph:
    def __init__(self, constituents: collections.abc.Iterable[BiltModelConstituent], backend: Backend) -> None:
        self.graph = nx.DiGraph()
        self._backend = backend
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
        sub_system = plan_step.operation
        if (
            isinstance(sub_system, SubSystemConstituent)
            and sub_system.required_inputs_for(plan_step.output, self._backend) is not None
        ):
            return
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
            raise BiltAnalysisError(msg)
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
        """Search for a path through a constituent node, resolving every input it needs."""
        match plan_step.operation:
            case SubSystemConstituent() as sub_system:
                return self._search_sub_system(plan_step, sub_system, available, visited)
            case _:
                wanted = plan_step.consumed
                resolved = self._resolve_inputs(wanted, available, visited)
                if len(resolved) != len(wanted):
                    return None
                return [*_flatten_sub_paths(resolved.values()), plan_step]

    def _search_sub_system(
        self,
        plan_step: PlanStep,
        sub_system: SubSystemConstituent,
        available: set[str],
        visited: set[Node],
    ) -> list[PlanStep] | None:
        """Search for a path through a sub-system, from the ports the parent can reach."""
        child_inputs = sub_system.required_inputs_for(plan_step.output, self._backend)
        if child_inputs is None:
            return None

        settled = dataclasses.replace(plan_step, child_inputs=child_inputs)
        resolved = self._resolve_inputs(settled.consumed, available, visited)
        if len(resolved) == len(settled.consumed):
            return [*_flatten_sub_paths(resolved.values()), settled]

        rest = [port for port in sorted(sub_system.__bilt_parameters__) if port not in resolved]
        reached = resolved | self._resolve_inputs(rest, available, visited)
        child_inputs = sub_system.required_inputs_for(plan_step.output, self._backend, within=reached)
        if child_inputs is None:
            return None

        settled = dataclasses.replace(plan_step, child_inputs=child_inputs)
        return [*_flatten_sub_paths(reached[port] for port in settled.consumed), settled]

    def _resolve_inputs(
        self,
        attributes: collections.abc.Iterable[Attribute],
        available: set[str],
        visited: set[Node],
    ) -> dict[Attribute, list[PlanStep]]:
        resolved: dict[Attribute, list[PlanStep]] = {}
        for attribute in attributes:
            sub_path = self.search(available, attribute, visited)
            if sub_path is not None:
                resolved[attribute] = sub_path
        return resolved

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