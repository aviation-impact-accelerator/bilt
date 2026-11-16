"""Bounded, cold structural-search benchmark for the nested coupled model."""

# ruff: noqa: INP001 - standalone benchmark companion, not an installed package

import dataclasses
import math
import typing
from contextlib import ExitStack
from unittest.mock import patch

from nested_coupled_models import EXPECTED_FINAL_POWER_W, RAW_INPUTS, build_nested_coupled_model

from bilt._backend._case import BackendCase
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._errors import BiltAnalysisError
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver import _planning
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    from bilt import Implementation
    from bilt._backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._plan import PlanBlock


class PlanSearchBudgetError(Exception):
    """Stop the diagnostic at the specified number of planning attempts."""


PLAN_ATTEMPT_BUDGET = 1_000


@dataclasses.dataclass
class NestedCoupledTrial:
    """One fresh implementation, context and bounded request."""

    coupled: bool
    implementation: Implementation
    request: EvaluationRequest
    resources: ExitStack
    result: BackendCase | PlanSearchBudgetError | BiltAnalysisError | None = None
    plan_attempts: int = 0
    checked: bool = False
    closed: bool = False

    @classmethod
    def prepare(cls, *, coupled: bool) -> typing.Self:
        """Prepare one independent trial and its evaluation context."""
        resources = ExitStack()
        try:
            implementation = build_nested_coupled_model(coupled=coupled)
            request = EvaluationRequest(
                output="final_power",
                input_domains=dict.fromkeys(RAW_INPUTS, Domain.scalar()),
            )
            resources.enter_context(EvaluationContext())
            return cls(coupled, implementation, request, resources)
        except BaseException:
            resources.close()
            raise

    def analyse(self) -> None:
        """Time the normal structural analysis, capped by planning work."""
        if self.result is not None or self.closed:
            msg = "Each trial must be analysed once."
            raise RuntimeError(msg)

        original_plan = _planning.plan_blocks

        def limited_plan(
            assignments: typing.Mapping[str, BiltModelConstituent],
            available: set[str],
            output: str,
            *,
            backend: Backend,
        ) -> tuple[PlanBlock, ...]:
            self.plan_attempts += 1
            if self.plan_attempts > PLAN_ATTEMPT_BUDGET:
                raise PlanSearchBudgetError
            return original_plan(assignments, available, output, backend=backend)

        try:
            with patch.object(_planning, "plan_blocks", limited_plan):
                self.result = self.implementation.analyser_for(PYTHON_BACKEND).analyse(self.request)
        except PlanSearchBudgetError as error:
            self.result = error
        except BiltAnalysisError as error:
            self.result = error

    def check(self) -> dict[str, object]:
        """Require a correct answer or, for the coupled case, the known work limit."""
        if isinstance(self.result, BackendCase):
            computed = PYTHON_BACKEND.chain_case(self.result).callable(**RAW_INPUTS)
            if not isinstance(computed, (int, float)) or not math.isclose(
                computed, EXPECTED_FINAL_POWER_W, rel_tol=1e-10
            ):
                msg = f"Expected {EXPECTED_FINAL_POWER_W}, got {computed!r}."
                raise AssertionError(msg)
            status = "solved"
        elif self.coupled and isinstance(self.result, PlanSearchBudgetError):
            status = "budget_exceeded"
        else:
            msg = f"Unexpected plan result: {self.result!r}."
            raise AssertionError(msg) from (self.result if isinstance(self.result, Exception) else None)

        self.checked = True
        return {
            "status": status,
            "formulation": "coupled" if self.coupled else "explicit",
            "candidate_budget": PLAN_ATTEMPT_BUDGET,
            "plan_attempts": self.plan_attempts,
            "expected_final_power_w": EXPECTED_FINAL_POWER_W,
        }

    def close(self) -> None:
        """Release the evaluation context even aoutput_stageer budget exhaustion."""
        self.resources.close()
        self.closed = True
