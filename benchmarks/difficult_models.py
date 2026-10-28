"""Different request directions and missing-reference searches in thermal networks."""

# ruff: noqa: INP001 - benchmark helper, not an installed package

import dataclasses
import typing

from scale_models import ScaleCase, ScaleTrial
from structural_search import SearchCase, check_coupled_rejection
from thermal_models import (
    ThermalPorts,
    heat_balance,
    left_capacity,
    left_flow_parallel,
    left_flow_series,
    left_heat,
    left_inlet,
    outlet_parallel,
    outlet_series,
    right_capacity,
    right_flow_parallel,
    right_flow_series,
    right_heat,
    right_inlet_parallel,
    right_inlet_series,
)

from bilt import Implementation
from bilt.core._errors import BiltAnalysisError
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltUnderconstrainedError
from bilt.symbolic._domains import Domain


@dataclasses.dataclass(frozen=True)
class DifficultCase:
    """A search challenge with an independent physical reference."""

    solver: str
    scenario: str
    depth: int


def build_mixed_model(depth: int) -> Implementation:
    """Alternate series and parallel layers, starting with a series pair."""
    implementation = Implementation(ThermalPorts, [heat_balance])
    common = (left_inlet, left_heat, right_heat, left_capacity, right_capacity)
    for level in range(depth):
        connections = (
            (right_inlet_series, left_flow_series, right_flow_series, outlet_series)
            if level % 2 == 0
            else (right_inlet_parallel, left_flow_parallel, right_flow_parallel, outlet_parallel)
        )
        implementation = Implementation(
            ThermalPorts,
            (*common, *connections),
            internal_implementations=dict.fromkeys(("left", "right"), implementation),
        )
    return implementation


@dataclasses.dataclass
class DifficultTrial(ScaleTrial):
    """Check successful answers or the specific absence of a temperature reference."""

    @classmethod
    def prepare_difficult(cls, case: DifficultCase) -> DifficultTrial:
        """Prepare a cold trial using the existing scoped solver/context lifecycle."""
        if case.scenario not in {"reverse", "heat", "flow", "mixed", "unanchored"}:
            raise ValueError(case)
        trial = typing.cast("DifficultTrial", cls.prepare_scale(ScaleCase(case.solver, "series", case.depth)))
        try:
            complete = {"inlet": 300.0, "outlet": 315.0, "heat": 120_000.0, "flow": 2.0, "capacity": 4_000.0}
            output = {"reverse": "inlet", "heat": "heat", "flow": "flow"}.get(case.scenario, "outlet")
            trial.values = {key: value for key, value in complete.items() if key != output}
            if case.scenario == "unanchored":
                # Neither endpoint temperature is supplied. Energy balance fixes
                # their difference, so absolute outlet temperature is undetermined.
                trial.values.pop("inlet")
            if case.scenario == "mixed":
                trial.implementation = build_mixed_model(case.depth)
            trial.expected = complete[output]
            trial.request = EvaluationRequest(output=output, input_domains=dict.fromkeys(trial.values, Domain.scalar()))
            trial.case = SearchCase(
                solver=case.solver, layout=case.scenario, coupled=False, depth=case.depth, rectangular=False
            )
        except BaseException:
            trial.close()
            raise
        else:
            return trial

    def check(self) -> dict[str, object]:
        """Check every round; unrelated isolation or compilation errors must fail."""
        if self.case.layout == "flow" or (self.case.layout == "heat" and self.case.solver != "scc"):
            # Shared heat/flow must be solved simultaneously with intermediate
            # temperatures. A unique physical solution need not have an acyclic plan.
            details = check_coupled_rejection(self.result, solver=self.case.solver, output=self.request.output)
            self.checked = True
            return details | {"scenario": self.case.layout, "target": self.request.output}
        if self.case.layout != "unanchored":
            return super().check() | {"scenario": self.case.layout, "target": self.request.output}
        result = self.result
        if self.case.solver == "scc":
            cause = result
            visited: set[int] = set()
            while isinstance(cause, BiltUnderconstrainedError) and id(cause) not in visited:
                visited.add(id(cause))
                if cause.__cause__ is None:
                    break
                cause = cause.__cause__
            valid = isinstance(cause, BiltUnderconstrainedError) and cause.__cause__ is None
        else:
            valid = (
                type(result) is BiltAnalysisError
                and str(result).startswith("Cannot compute outlet from available inputs ")
                and result.__cause__ is None
                and "route tried was rejected" not in str(result)
            )
        if not valid:
            msg = f"Expected missing-temperature-reference rejection, received {result!r}."
            raise AssertionError(msg) from (result if isinstance(result, Exception) else None)
        self.checked = True
        return {
            "status": "rejected",
            "scenario": self.case.layout,
            "target": self.request.output,
            "error_type": type(result).__name__,
            "rejection_reason": "underconstrained" if self.case.solver == "scc" else "no_acyclic_plan",
        }
