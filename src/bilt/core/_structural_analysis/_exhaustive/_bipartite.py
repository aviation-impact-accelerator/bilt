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
    def __init__(self, constituents: collections.abc.Collection[BiltModelConstituent]) -> None:
        self.graph = nx.DiGraph()
        for constituent in constituents:
            match constituent:
                case BasicTransform():
                    transform_name = constituent.__bilt_name__ + "_transform"
                    self.graph.add_node(
                        transform_name, bipartite="constituent", obj=constituent, target=constituent.__bilt_name__
                    )
                    self.graph.add_node(constituent.__bilt_name__, bipartite="attribute")
                    self.graph.add_edge(transform_name, constituent.__bilt_name__)
                    for param in constituent.__bilt_parameters__:
                        self.graph.add_node(param, bipartite="attribute")
                        self.graph.add_edge(param, transform_name)
                case BasicEquation():
                    for param in constituent.__bilt_parameters__:
                        self.graph.add_node(param, bipartite="attribute")

                    for param in constituent.__bilt_parameters__:
                        equation_name = constituent.__bilt_name__ + "_" + param
                        self.graph.add_node(equation_name, bipartite="constituent", obj=constituent, target=param)
                        self.graph.add_edge(equation_name, param)
                        for param2 in constituent.__bilt_parameters__:
                            if param2 != param:
                                self.graph.add_edge(param2, equation_name)
        if not self.is_bipartite():
            msg = "The graph is not bipartite. Please check the constituents provided."
            raise ValueError(msg)

    def is_bipartite(self) -> bool:
        return nx.is_bipartite(self.graph)

    def remove_failed_plan_step(
        self,
        plan_step: PlanStep,
    ) -> None:
        """If an System or Equation is not able to be used in a certain direction, remove it from the graph."""
        self.graph.remove_node(plan_step.operation.__bilt_name__ + "_" + plan_step.output)

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
                    inputs=frozenset(predecessors),
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
