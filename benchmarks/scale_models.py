"""Cold planning trials with explicit graph sizes and analytic scalar answers."""

# ruff: noqa: INP001 - benchmark helper, not an installed package

import dataclasses
import math
from contextlib import ExitStack
from unittest.mock import patch

from structural_models import build_model, expected_value
from structural_search import SearchCase, SearchTrial, scc
from thermal_models import build_thermal_model

from bilt._backend._case import BackendCase
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import StructuralExpansion
from bilt.core._structural_analysis._exhaustive import ExhaustiveStructuralAnalyser
from bilt.core._system import _implementation as implementation_module
from bilt.symbolic._domains import Domain


@dataclasses.dataclass(frozen=True)
class ScaleCase:
    """A topology and depth; actual expanded graph counts accompany results."""

    solver: str
    topology: str
    depth: int


@dataclasses.dataclass
class ScaleTrial(SearchTrial):
    """Reuse the timed search lifecycle with custom inputs and reference values."""

    values: dict[str, float] = dataclasses.field(default_factory=dict)
    expected: float = 0.0

    @classmethod
    def prepare_scale(cls, case: ScaleCase) -> ScaleTrial:
        """Create the model and scoped solver selection before timing."""
        if case.solver not in {"exhaustive", "scc"} or case.depth < 0:
            raise ValueError(case)
        resources = ExitStack()
        try:
            factory = (
                ExhaustiveStructuralAnalyser if case.solver == "exhaustive" else scc.CausalizationStructuralAnalyzer
            )
            resources.enter_context(patch.object(implementation_module, "CausalizationStructuralAnalyzer", factory))
            if case.topology in {"chain", "tree"}:
                layout = "links" if case.topology == "chain" else "siblings"
                implementation = build_model(coupled=False, depth=case.depth, layout=layout)
                values = {"a": 4.0, "b": 2.0}
                expected = expected_value(coupled=False, depth=case.depth, layout=layout)
                output = "x"
            else:
                implementation = build_thermal_model(case.topology, case.depth)
                values = {"inlet": 300.0, "heat": 120_000.0, "flow": 2.0, "capacity": 4_000.0}
                # Independent whole-network energy balance, not a recursive solver.
                expected = values["inlet"] + values["heat"] / (values["flow"] * values["capacity"])
                output = "outlet"
            request = EvaluationRequest(output=output, input_domains=dict.fromkeys(values, Domain.scalar()))
            resources.enter_context(EvaluationContext())
            return cls(
                SearchCase(
                    solver=case.solver, layout=case.topology, coupled=False, depth=case.depth, rectangular=False
                ),
                implementation,
                request,
                resources,
                values=values,
                expected=expected,
            )
        except BaseException:
            resources.close()
            raise

    def check(self) -> dict[str, object]:
        """Verify the numerical answer and record the expanded workload size."""
        if not isinstance(self.result, BackendCase):
            msg = f"Expected a successful scaling plan, received {self.result!r}."
            raise AssertionError(msg) from (self.result if isinstance(self.result, Exception) else None)  # noqa: TRY004 - benchmark correctness assertion
        value = PYTHON_BACKEND.chain_case(self.result).callable(**self.values)
        if not isinstance(value, (float, int)) or not math.isclose(value, self.expected, rel_tol=1e-12, abs_tol=1e-12):
            msg = f"Expected {self.expected}, received {value}."
            raise AssertionError(msg)
        expansion = StructuralExpansion(
            self.implementation.structural_analysis()._incidence,  # noqa: SLF001
            set(self.request.input_domains),
            self.request.output,
        )
        graph, attributes = expansion.collect()
        snapshot = expansion.snapshot()
        self.checked = True
        return {
            "status": "success",
            "value": value,
            "expected_value": self.expected,
            "constraints": len(graph) - len(attributes),
            "unknowns": len(attributes),
            "subsystem_instances": len(snapshot.expanded_paths),
            "leaf_instances": 1 if self.case.layout == "chain" else 2**self.case.depth,
            "depth": self.case.depth,
            "topology": self.case.layout,
        }
