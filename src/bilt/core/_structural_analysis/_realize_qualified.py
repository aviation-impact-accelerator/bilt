"""Realise qualified pointwise plans using the existing primitive and linear compilers."""

import dataclasses
import typing

from bilt._backend._callable import BackendCallable
from bilt._backend._case import BackendCase
from bilt.core._errors import BiltAnalysisError
from bilt.core._port import mangle_name
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._structural_analysis._qualified_block import QualifiedBlock, QualifiedPlan, QualifiedStep
from bilt.core._structural_analysis._realize import RealizationFailure, _operation_to_callable, _output_domain
from bilt.core._structural_analysis._realize_linear import realize_linear_block
from bilt.core._tensors._errors import BiltSymbolicLoweringError, BiltTensorDomainError

if typing.TYPE_CHECKING:
    from bilt._backend import Backend
    from bilt.symbolic._domains import Domain


@dataclasses.dataclass(frozen=True)
class QualifiedRealizationFailure:
    """A rejected primitive direction or complete qualified block formulation."""

    failed: QualifiedStep | QualifiedBlock
    cause: Exception


type QualifiedFailureKey = tuple[QualifiedBlock, frozenset[tuple[str, Domain]]]


def realize_qualified_plan(
    plan: QualifiedPlan,
    backend: Backend,
    block_failures: dict[QualifiedFailureKey, QualifiedRealizationFailure],
) -> BackendCase | QualifiedRealizationFailure:
    """Keep block rejections request-local and preserve qualified step identity."""
    domains = plan.request.input_domains.copy()
    intermediates = []
    reserved = set(map(mangle_name, plan.outputs | plan.request.input_domains.keys()))
    for index, block in enumerate(plan.blocks):
        if isinstance(block, QualifiedBlock):
            key = block, frozenset((name, domains[name]) for name in block.inputs)
            if key in block_failures:
                return block_failures[key]
            name = f"__bilt_qualified_solution_{index}"
            while name in reserved:
                name += "_"
            try:
                callables = realize_linear_block(block, backend, domains, solution_name=name)
            except (BiltLinearBlockError, BiltSymbolicLoweringError, BiltTensorDomainError) as error:
                sources = ", ".join(row.label for row in block.rows)
                cause = BiltLinearBlockError(f"Qualified block from [{sources}] was rejected: {error}")
                cause.__cause__ = error
                failure = QualifiedRealizationFailure(block, cause)
                block_failures[key] = failure
                return failure
            intermediates.extend(callables)
            for output, callable_ in zip(block.unknowns, callables[1:], strict=True):
                domains[output] = _output_domain(callable_, output)
        else:
            result = _realize_step(block, backend, domains)
            if isinstance(result, QualifiedRealizationFailure):
                return result
            intermediates.append(result)
            domains[block.output] = _output_domain(result, block.output)
    return BackendCase(
        name=mangle_name(plan.request.output),
        parameters=tuple(map(mangle_name, sorted(plan.request.input_domains))),
        intermediates=intermediates,
        return_name=mangle_name(plan.request.output),
    )


def _realize_step(
    step: QualifiedStep, backend: Backend, domains: dict[str, Domain]
) -> BackendCallable | QualifiedRealizationFailure:
    local_domains = {
        name: domains[step.prefix + name] for name in step.source.__bilt_parameters__ if step.prefix + name in domains
    }
    result = _operation_to_callable(
        PlanStep(step.source, step.output[len(step.prefix) :]), backend, symbol_domains=local_domains
    )
    if isinstance(result, RealizationFailure):
        cause = BiltAnalysisError(
            f"Cannot realise {step.output!r} using {step.prefix + step.source.__bilt_name__!r}: {result.cause}"
        )
        cause.__cause__ = result.cause
        return QualifiedRealizationFailure(step, cause)
    names = {mangle_name(name): mangle_name(step.prefix + name) for name in step.source.__bilt_parameters__}
    parent_to_local = {names[name]: name for name in result.parameters}

    def evaluate(*args: object, **kwargs: object) -> object:
        return result.callable(*args, **{parent_to_local[name]: value for name, value in kwargs.items()})

    return BackendCallable(
        name=mangle_name(step.output),
        parameters=tuple(names[name] for name in result.parameters),
        callable=evaluate,
        output_domain=result.output_domain,
    )
