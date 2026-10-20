"""Cold structural-search trials and optional untimed search instrumentation."""

# ruff: noqa: INP001 - benchmark helpers are not an installed package

import dataclasses
import functools
import typing
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch

from structural_models import build_model, expected_value

from bilt._backend._case import BackendCase
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._errors import BiltAnalysisError
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver import _backend as scc
from bilt.core._structural_analysis._causalization_block_solver._dm import dulmage_mendelsohn
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltOverconstrainedError,
    BiltUnderconstrainedError,
)
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import StructuralExpansion
from bilt.core._structural_analysis._exhaustive import _backend as exhaustive
from bilt.core._structural_analysis._exhaustive._backend import ExhaustiveStructuralAnalyser
from bilt.core._structural_analysis._exhaustive._bipartite import BipartiteGraph
from bilt.core._system import _implementation as implementation_module
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from bilt import Implementation
    from bilt.core._model_constituent import BiltModelConstituent


@dataclasses.dataclass(frozen=True)
class SearchCase:
    """One cell of the existing structural-search matrix."""

    solver: str
    layout: str
    coupled: bool
    depth: int
    rectangular: bool


@dataclasses.dataclass
class SearchTrial:
    """One fresh model, solver selection and evaluation context."""

    case: SearchCase
    implementation: Implementation
    request: EvaluationRequest
    resources: ExitStack
    result: BackendCase | BiltAnalysisError | None = None
    checked: bool = False
    closed: bool = False

    @classmethod
    def prepare(cls, case: SearchCase) -> typing.Self:
        """Build fixtures and enter their contexts before timing begins."""
        resources = ExitStack()
        try:
            factory = (
                ExhaustiveStructuralAnalyser if case.solver == "exhaustive" else scc.CausalizationStructuralAnalyzer
            )
            resources.enter_context(patch.object(implementation_module, "ExhaustiveStructuralAnalyser", factory))
            implementation = build_model(
                coupled=case.coupled, depth=case.depth, rectangular=case.rectangular, layout=case.layout
            )
            request = EvaluationRequest(output="x", input_domains={"a": Domain.scalar(), "b": Domain.scalar()})
            resources.enter_context(EvaluationContext())
            return cls(case, implementation, request, resources)
        except BaseException:
            resources.close()
            raise

    def analyse(self) -> None:
        """Time analyser construction, planning and realisation exactly once."""
        if self.closed or self.result is not None:
            msg = "Each benchmark trial must be analysed once in a fresh context."
            raise RuntimeError(msg)
        try:
            self.result = self.implementation.analyser_for(PYTHON_BACKEND).analyse(self.request)
        except BiltAnalysisError as error:
            self.result = error

    def check(self) -> dict[str, object]:
        """Verify every round's answer or rejection outside the timer."""
        result = self.result
        details: dict[str, object]
        if self.case.coupled:
            if not isinstance(result, BiltAnalysisError) or isinstance(
                result, (BiltUnderconstrainedError, BiltOverconstrainedError)
            ):
                msg = f"Expected unsupported coupled execution, received {result!r}."
                raise AssertionError(msg)
            details = {"status": "rejected", "error_type": type(result).__name__, "message": str(result)}
        else:
            if not isinstance(result, BackendCase):
                msg = f"Expected a successful plan, received {result!r}."
                raise AssertionError(msg) from (result if isinstance(result, Exception) else None)
            value = PYTHON_BACKEND.chain_case(result).callable(a=4.0, b=2.0)
            expected = expected_value(coupled=False, depth=self.case.depth, layout=self.case.layout)
            if value != expected:
                msg = f"Expected {expected}, received {value}."
                raise AssertionError(msg)
            details = {"status": "success", "value": value, "expected_value": expected}
        if self.case.rectangular:
            details["structural_shape"] = self._check_rectangular_shape()
        self.checked = True
        return details

    def _check_rectangular_shape(self) -> dict[str, int]:
        incidence = self.implementation.structural_analysis()._incidence  # noqa: SLF001
        graph, attributes = StructuralExpansion(
            incidence, set(self.request.input_domains), self.request.output
        ).collect()
        partition = dulmage_mendelsohn(graph, frozenset(attributes.values()))
        rows = len(graph) - len(attributes)
        if rows >= len(attributes) or attributes[self.request.output] not in partition.well_determined:
            msg = "Rectangular benchmark must have a well-determined target and surplus unknowns."
            raise AssertionError(msg)
        return {
            "constraints": rows,
            "unknowns": len(attributes),
            "well_determined_unknowns": len(set(attributes.values()) & partition.well_determined),
            "underdetermined_unknowns": len(set(attributes.values()) & partition.underdetermined),
        }

    def close(self) -> None:
        """Restore the solver selection and release the evaluation context."""
        self.resources.close()
        self.closed = True


def instrument(solver: str) -> tuple[ExitStack, Counter[str]]:
    """Count search operations without adding overhead to timing-only trials."""
    stack = ExitStack()
    counts: Counter[str] = Counter()
    queries: set[tuple[int, frozenset[str], str]] = set()
    factory = ExhaustiveStructuralAnalyser if solver == "exhaustive" else scc.CausalizationStructuralAnalyzer
    original_required = typing.cast(
        "Callable[[object, set[str], str], frozenset[str] | None]", factory.required_inputs_for_target
    )

    def required(self: object, available: set[str], output: str) -> frozenset[str] | None:
        counts["child_route_queries"] += 1
        key = id(self), frozenset(available), output
        if key not in queries:
            queries.add(key)
            counts["distinct_child_route_queries"] += 1
        return original_required(self, available, output)

    def counted(name: str, function: Callable[..., object]) -> Callable[..., object]:
        @functools.wraps(function)
        def call(*args: object, **kwargs: object) -> object:
            counts[name] += 1
            return function(*args, **kwargs)

        return call

    stack.enter_context(patch.object(factory, "required_inputs_for_target", required))
    if solver == "scc":
        for name, label in (
            ("find_matching", "execution_matching_calls"),
            ("settle_assignments", "settlement_calls"),
            ("realize_plan", "realisation_attempts"),
            ("validate_request_structure", "structural_validation_calls"),
        ):
            stack.enter_context(patch.object(scc, name, counted(label, getattr(scc, name))))
        original_plan = typing.cast(
            "Callable[[object, dict[str, BiltModelConstituent], set[str], str], object]",
            scc.CausalizationStructuralAnalyzer._plan_steps,  # noqa: SLF001 - benchmark instrumentation
        )

        def plan(
            self: object, assignments: dict[str, BiltModelConstituent], available: set[str], output: str
        ) -> object:
            counts["candidate_checks"] += 1
            if output not in available and output not in assignments:
                counts["candidates_without_target_assignment"] += 1
            return original_plan(self, assignments, available, output)

        stack.enter_context(patch.object(factory, "_plan_steps", plan))
    else:
        stack.enter_context(
            patch.object(BipartiteGraph, "search", counted("recursive_search_calls", BipartiteGraph.search))
        )
        stack.enter_context(
            patch.object(exhaustive, "realize_plan", counted("realisation_attempts", exhaustive.realize_plan))
        )
    return stack, counts


def count_search(case: SearchCase) -> dict[str, object]:
    """Run instrumentation on a separate fresh trial after timing finishes."""
    trial = SearchTrial.prepare(case)
    try:
        resources, counts = instrument(case.solver)
        with resources:
            trial.analyse()
        details = trial.check()
        details["counts"] = dict(counts)
        return details
    finally:
        trial.close()


if __name__ == "__main__":
    msg = "Run structural benchmarks with: uv run pytest benchmarks --benchmark-only --timeout=100"
    raise SystemExit(msg)
