__all__ = ["BipartiteGraph"]

import collections.abc
import dataclasses
import typing

import networkx as nx

from bilt.core._equation import BasicEquation
from bilt.core._model_constituent import BiltModelConstituent
from bilt.core._structural_analysis._errors import BiltSystemAnalysisError
from bilt.core._structural_analysis._plan import (
    AnalysisPlan,
    CoupledBlock,
    PlanItem,
    PlanStep,
)
from bilt.core._structural_analysis._scc._dependencies import (
    required_inputs_for_step,
    sub_system_plan_for_step,
)
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import BasicTransform

type Attribute = str
type Constituent = BasicEquation | BasicTransform | SubSystemConstituent
type DependencyNode = Attribute | PlanStep


class BipartiteGraph:
    """Incidence graph between model constituents and attributes."""

    def __init__(
        self,
        constituents: collections.abc.Iterable[BiltModelConstituent],
    ) -> None:
        self.graph = nx.Graph()

        for constituent in constituents:
            self._add_constituent(constituent)

    def find_plan(
        self,
        available: set[str],
        output: str,
        *,
        excluded: frozenset[PlanStep] = frozenset(),
    ) -> AnalysisPlan | None:
        """Find a structural plan that produces output from available inputs."""
        if output not in self.graph:
            msg = f"Output {output!r} is not in the graph."
            raise BiltSystemAnalysisError(msg)

        if output in available:
            return AnalysisPlan(
                inputs=frozenset(available),
                output=output,
                steps=(),
            )

        matching = self._target_matching(
            available,
            output,
            excluded,
        )

        if matching is None:
            return None

        dependency_graph = self._dependency_graph(matching.values())

        return AnalysisPlan(
            inputs=frozenset(available),
            output=output,
            steps=self._ordered_plan_items(dependency_graph),
        )

    def _plan_step(
        self,
        operation: Constituent,
        output: str,
        available: set[str],
    ) -> PlanStep | None:
        """Create a plan step, retaining a selected child route."""
        step = PlanStep(
            operation=operation,
            output=output,
        )

        if not isinstance(operation, SubSystemConstituent):
            return step

        available_child_ports = self._available_child_ports(
            operation,
            output,
            available,
        )

        sub_system_plan = sub_system_plan_for_step(
            step,
            available_child_ports=available_child_ports,
        )

        if sub_system_plan is None:
            return None

        return dataclasses.replace(
            step,
            sub_system_plan=sub_system_plan,
        )

    def _available_child_ports(
        self,
        sub_system: SubSystemConstituent,
        output: str,
        available: set[str],
    ) -> set[str]:
        """Return child ports that the parent can potentially provide."""
        child_output = sub_system.unqualify(output)

        if child_output is None:
            msg = f"{sub_system!r} does not have port {output!r}."
            raise BiltSystemAnalysisError(msg)

        child_ports: set[str] = set()

        for child_port in sub_system.interface.__bilt_parameters__:
            if child_port == child_output:
                continue

            qualified_port = sub_system.qualify(child_port)

            if qualified_port in available:
                child_ports.add(child_port)
                continue

            if self._has_parent_producer(
                sub_system,
                qualified_port,
            ):
                child_ports.add(child_port)

        return child_ports

    def _has_parent_producer(
        self,
        sub_system: SubSystemConstituent,
        attribute: str,
    ) -> bool:
        """Return whether the parent graph can produce a child port."""
        for neighbour in self.graph.neighbors(attribute):
            if neighbour is sub_system:
                continue

            if self.graph.edges[attribute, neighbour]["can_produce"]:
                return True

        return False

    def _add_constituent(
        self,
        constituent: BiltModelConstituent,
    ) -> None:
        match constituent:
            case BasicTransform() as transform:
                self._add_transform(transform)

            case BasicEquation() as equation:
                self._add_relation(equation)

            case SubSystemConstituent() as sub_system:
                self._add_relation(sub_system)

            case _:
                msg = f"Constituent type {type(constituent)} is not supported in the bipartite graph."
                raise NotImplementedError(msg)

    def _add_relation(
        self,
        relation: BasicEquation | SubSystemConstituent,
    ) -> None:
        """Add an acausal relation."""
        self.graph.add_node(
            relation,
            bipartite="constituent",
        )

        for parameter in relation.__bilt_parameters__:
            self.graph.add_node(
                parameter,
                bipartite="attribute",
            )
            self.graph.add_edge(
                relation,
                parameter,
                can_produce=True,
            )

    def _add_transform(
        self,
        transform: BasicTransform,
    ) -> None:
        """Add a causal transform."""
        self.graph.add_node(
            transform,
            bipartite="constituent",
        )

        self.graph.add_node(
            transform.__bilt_name__,
            bipartite="attribute",
        )
        self.graph.add_edge(
            transform,
            transform.__bilt_name__,
            can_produce=True,
        )

        for parameter in transform.__bilt_parameters__:
            self.graph.add_node(
                parameter,
                bipartite="attribute",
            )
            self.graph.add_edge(
                transform,
                parameter,
                can_produce=False,
            )

    def _target_matching(
        self,
        available: set[str],
        output: str,
        excluded: frozenset[PlanStep],
    ) -> dict[str, PlanStep] | None:
        """Find a matching sufficient to produce the requested output."""
        prioritised = {output}
        seen: set[frozenset[str]] = set()

        while True:
            key = frozenset(prioritised)

            if key in seen:
                return None

            seen.add(key)

            matching = self._preferred_matching(
                available,
                prioritised,
                excluded,
            )

            required, missing = self._dependency_closure(
                matching,
                available,
                output,
            )

            if not missing:
                return {attribute: matching[attribute] for attribute in required}

            prioritised = required | missing

    def _preferred_matching(
        self,
        available: set[str],
        prioritised: set[str],
        excluded: frozenset[PlanStep],
    ) -> dict[str, PlanStep]:
        """Find a matching that strongly prefers required attributes."""
        graph = nx.Graph()
        candidate_steps: dict[
            tuple[Constituent, Attribute],
            PlanStep,
        ] = {}

        attribute_count = sum(1 for _, data in self.graph.nodes(data=True) if data["bipartite"] == "attribute")

        priority_weight = attribute_count + 1

        for left, right, data in self.graph.edges(data=True):
            if self.graph.nodes[left]["bipartite"] == "attribute":
                attribute = typing.cast("Attribute", left)
                operation = typing.cast("Constituent", right)
            else:
                operation = typing.cast("Constituent", left)
                attribute = typing.cast("Attribute", right)

            if attribute in available:
                continue

            if not data["can_produce"]:
                continue

            step = self._plan_step(
                operation,
                attribute,
                available,
            )

            if step is None:
                continue

            if step in excluded:
                continue

            if required_inputs_for_step(step) is None:
                continue

            candidate_steps[(operation, attribute)] = step

            graph.add_edge(
                operation,
                attribute,
                weight=(priority_weight if attribute in prioritised else 1),
            )

        pairs = nx.max_weight_matching(
            graph,
            maxcardinality=False,
            weight="weight",
        )

        matching: dict[str, PlanStep] = {}

        for left, right in pairs:
            if isinstance(left, str):
                attribute = left
                operation = typing.cast("Constituent", right)
            else:
                operation = typing.cast("Constituent", left)
                attribute = typing.cast("Attribute", right)

            matching[attribute] = candidate_steps[(operation, attribute)]

        return matching

    def _dependency_closure(
        self,
        matching: dict[str, PlanStep],
        available: set[str],
        output: str,
    ) -> tuple[set[str], set[str]]:
        """Find the matched attributes needed to produce output.

        Returns the required matched attributes and any unresolved attributes
        that need to be prioritised by the next matching attempt.
        """
        required: set[str] = set()
        missing: set[str] = set()
        visited: set[str] = set()
        pending = [output]

        while pending:
            attribute = pending.pop()

            if attribute in available or attribute in visited:
                continue

            visited.add(attribute)

            step = matching.get(attribute)

            if step is None:
                missing.add(attribute)
                continue

            required.add(attribute)

            inputs = required_inputs_for_step(step)

            if inputs is None:
                missing.add(attribute)
                continue

            pending.extend(inputs)

        return required, missing

    def _dependency_graph(
        self,
        steps: collections.abc.Iterable[PlanStep],
    ) -> nx.DiGraph:
        """Build the causal dependency graph implied by a matching."""
        graph = nx.DiGraph()

        for step in steps:
            graph.add_node(step)
            graph.add_node(step.output)
            graph.add_edge(step, step.output)

            inputs = required_inputs_for_step(step)

            if inputs is None:
                msg = f"Cannot determine the inputs required by {step!r}."
                raise BiltSystemAnalysisError(msg)

            for input_ in inputs:
                graph.add_node(input_)
                graph.add_edge(input_, step)

        return graph

    def _ordered_plan_items(
        self,
        graph: nx.DiGraph,
    ) -> tuple[PlanItem, ...]:
        """Return explicit steps and coupled blocks in dependency order."""
        components = list(nx.strongly_connected_components(graph))

        condensation = nx.condensation(
            graph,
            components,
        )

        items: list[PlanItem] = []

        for component in nx.topological_sort(condensation):
            members = condensation.nodes[component]["members"]

            steps = sorted(
                (node for node in members if isinstance(node, PlanStep)),
                key=lambda step: step.output,
            )

            match steps:
                case []:
                    continue

                case [step]:
                    items.append(step)

                case _:
                    items.append(
                        CoupledBlock(
                            steps=tuple(steps),
                        )
                    )

        return tuple(items)
