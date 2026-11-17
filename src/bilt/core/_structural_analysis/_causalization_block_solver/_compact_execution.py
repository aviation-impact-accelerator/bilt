"""Experimental bridge from compact tensor matching to existing linear solves.

Only one primitive equation and one explicitly domain-hinted tensor unknown
are eligible. A compact *complete* dependency pattern and matching scalar
counts are required before the existing linear-operator compiler is used.
Other models retain the normal structural search and realization paths.
"""

import typing

from bilt._backend._case import BackendCase
from bilt.core._equation import BasicEquation
from bilt.core._evaluation_context import current_evaluation
from bilt.core._port import mangle_name
from bilt.core._structural_analysis._compact_incidence import (
    BiltCompactIncidenceError,
    build_compact_incidence,
)
from bilt.core._structural_analysis._compact_matching import match_compact_incidence
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.core._structural_analysis._linear_domains import equation_domain_hints
from bilt.core._structural_analysis._plan import PlanBlock, PlanStep
from bilt.core._structural_analysis._realize_linear import realize_linear_block
from bilt.core._tensors._errors import BiltSymbolicLoweringError, BiltTensorDomainError
from bilt.symbolic._operator_semantics import BiltOperatorSemanticsError

if typing.TYPE_CHECKING:
    from bilt._backend import Backend
    from bilt.core._evaluation_request import EvaluationRequest
    from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph


def try_compact_linear_execution(
    incidence: FullIncidenceGraph,
    request: EvaluationRequest,
    backend: Backend,
) -> BackendCase | None:
    """Use the normal joint solver for one structurally square tensor operator.

    Deliberately avoid changing route selection for multiple equations,
    subsystems, transforms, or unknowns without explicit domain hints.
    """
    operation = _single_equation(incidence, request)
    if operation is None:
        return None

    try:
        hints = dict(equation_domain_hints(operation))
        output_domain = hints.get(request.output)
        if output_domain is None:
            return None
        if any(
            name in hints and set(hints[name].dims) != set(domain.dims)
            for name, domain in request.input_domains.items()
        ):
            return None
        compact = build_compact_incidence(
            (operation.__bilt_equation_tree__,),
            {request.output: output_domain},
            request.input_domains,
        )
        matched = match_compact_incidence(compact, current_evaluation().domains, max_nodes=4096, max_edges=100_000)
        if matched.pattern != "complete" or not matched.well_determined:
            return None
        block = PlanBlock((PlanStep(operation, request.output),), is_cyclic=True)
        callables = realize_linear_block(
            block,
            backend,
            request.input_domains,
            solution_name="__bilt_compact_linear_solution",
            allow_distinct_row_domains=True,
        )
    except (
        BiltCompactIncidenceError,
        BiltOperatorSemanticsError,
        BiltLinearBlockError,
        BiltSymbolicLoweringError,
        BiltTensorDomainError,
    ):
        # This optimization must not turn a previously supported request
        # into a failure or replace the normal route/retry diagnostics.
        return None

    output = mangle_name(request.output)
    return BackendCase(
        name=output,
        parameters=tuple(map(mangle_name, sorted(request.input_domains))),
        intermediates=list(callables),
        return_name=output,
    )


def _single_equation(incidence: FullIncidenceGraph, request: EvaluationRequest) -> BasicEquation | None:
    """Require a self-contained primitive equation with exactly one unknown."""
    if request.output in request.input_domains or len(incidence.constituent_nodes) != 1:
        return None
    operation = incidence.graph[incidence.constituent_nodes[0]]
    if isinstance(operation, BasicEquation) and set(operation.__bilt_parameters__) == (
        set(request.input_domains) | {request.output}
    ):
        return operation
    return None
