from typing import TYPE_CHECKING

import networkx as nx

from bilt.core._equation import BasicEquation
from bilt.core._structural_analysis._errors import BiltSystemEvaluationError
from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._transform import BasicTransform

if TYPE_CHECKING:
    import collections.abc

    from bilt.core._model_constituent import BiltModelConstituent


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
            equation_name = equation.__bilt_name__ + "_" + param
            self.graph.add_node(equation_name, bipartite="constituent", obj=equation, target=param)
            self.graph.add_edge(equation_name, param)
            for param2 in equation.__bilt_parameters__:
                if param2 != param:
                    self.graph.add_edge(param2, equation_name)

    def _add_transform(self, transform: BasicTransform) -> None:
        """Add a new transform to the graph."""
        transform_name = transform.__bilt_name__ + "_transform"
        self.graph.add_node(transform_name, bipartite="constituent", obj=transform, target=transform.__bilt_name__)
        self.graph.add_node(transform.__bilt_name__, bipartite="attribute")
        self.graph.add_edge(transform_name, transform.__bilt_name__)
        for param in transform.__bilt_parameters__:
            self.graph.add_node(param, bipartite="attribute")
            self.graph.add_edge(param, transform_name)

    def _remove_equation(self, equation: BasicEquation, output: str) -> None:
        """Remove an equation from the graph."""
        self.graph.remove_node(equation.__bilt_name__ + "_" + output)

    def _remove_transform(self, transform: BasicTransform) -> None:
        """Remove a transform from the graph."""
        self.graph.remove_node(transform.__bilt_name__ + "_transform")

    def remove_failed_plan_step(
        self,
        plan_step: PlanStep,
    ) -> None:
        """If an System or Equation is found to be unable to be used in a certain direction after a realization, remove it from the graph."""
        match plan_step.operation:
            case BasicTransform() as transform:
                self._remove_transform(transform)
            case BasicEquation() as equation:
                self._remove_equation(equation, plan_step.output)

    def search(self, available: set[str], output: str, visited: list) -> list[PlanStep] | None:
        """Perform a depth-first search to find a path from available inputs to the desired output."""
        if output not in self.graph:
            msg = f"Output {output} is not in the graph. Please check the constituents provided."
            raise BiltSystemEvaluationError(msg)
        if output in available:
            return []
        if output in visited:
            return None

        predecessors = list(self.graph.predecessors(output))
        next_visited = [*visited, output]

        if self.graph.nodes[output]["bipartite"] == "constituent":
            path: list[PlanStep] = []
            for neighbor in predecessors:
                sub_path = self.search(available, neighbor, next_visited)
                if sub_path is None:
                    return None
                path.extend(step for step in sub_path if step not in path)
            path.append(
                PlanStep(
                    operation=self.graph.nodes[output]["obj"],
                    output=self.graph.nodes[output]["target"],
                )
            )
            return path

        for neighbor in predecessors:
            sub_path = self.search(available, neighbor, next_visited)
            if sub_path is not None:
                return sub_path
        return None
