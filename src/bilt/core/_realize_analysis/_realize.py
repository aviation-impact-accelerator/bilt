__all__ = ["RealizationFailure", "realize_plan"]

import ast
import dataclasses
import inspect
import textwrap
import typing

from bilt._backend._callable import BackendCallable
from bilt._backend._case import BackendCase
from bilt.core._equation import BasicEquation
from bilt.core._interface import Interface
from bilt.core._tensors._tensor_analysis import DomainedVar, analyse_domains_in_signature, canonicalised, clear_signature_type_hints
from bilt.core._transform import BasicTransform
from bilt.symbolic._domains import union
from bilt.symbolic._isolation._errors import IsolationError
from bilt.symbolic._isolation._isolate import equation_to_ast

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend
    from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep


def realize_plan(plan: AnalysisPlan, backend: Backend) -> BackendCase | RealizationFailure:
    """Convert an AnalysisPlan into a BackendCase by compiling each step to a callable."""
    intermediates: list[BackendCallable] = []
    parameters: dict[str, DomainedVar] = {}
    intermediate_opts: dict[str, DomainedVar] = {}

    for step in plan.steps:
        result = _operation_to_callable(step, backend=backend)

        if isinstance(result, RealizationFailure):
            return result
        
        parameters |= {
            domained_var.name: domained_var
            for domained_var in result.expected_parameters.values()
        }

        intermediate_opts[step.output] = DomainedVar(step.output, result.expected_return_domain)
        intermediates.append(result)

    # Assuming no reduction
    output_domain = canonicalised(union(*[var.domain.domain for var in parameters.values()],
                                        *[var.domain.domain for var in intermediate_opts.values()]))

    return BackendCase(
        name=plan.output,
        parameters={name: parameters[name] for name in plan.inputs},
        intermediates={intermediate.name: intermediate for intermediate in intermediates},
        returns=DomainedVar(plan.output, output_domain),
    )


def _operation_to_callable(plan_step: PlanStep, backend: Backend) -> BackendCallable | RealizationFailure:
    match plan_step.operation:
        case BasicTransform() as transform:
            function_ast = typing.cast(
                "ast.FunctionDef",
                ast.parse(textwrap.dedent(inspect.getsource(transform._function))).body[0],  # noqa: SLF001
            )
            function_ast.decorator_list = [
                d for d in function_ast.decorator_list if not (isinstance(d, ast.Name) and d.id == "transform")
            ]

            # TODO: Nic said we should be doing this after ast transforming in transpile... but having
            # code to insert the alignment prelude, when that is only relevant when transpiling an
            # operation to callable and not for a chain case, *in* transpile, seems silly.
            analysis = analyse_domains_in_signature(function_ast)
            clear_signature_type_hints(function_ast)

            callable = backend.transpile(function_ast)

            return BackendCallable(
                name=function_ast.name,
                expected_parameters={param.name: param for param in analysis.parameters.values()},
                expected_return_domain=analysis.return_domain,
                expected_union_domain=analysis.union_domain,
                callable=callable,
            )
        case BasicEquation() as equation:
            try:
                function_ast = equation_to_ast(equation, plan_step.output)
                msg = "Equation tensor analysis is not yet implemented."
                raise NotImplementedError(msg)
                return backend.transpile(function_ast)
            except IsolationError:
                return RealizationFailure(failed_plan_step=plan_step)
        case Interface():
            msg = "Interface evaluation is not yet implemented."
            raise NotImplementedError(msg)
        case _:
            msg = f"Operation type {type(plan_step.operation)} is not supported for evaluation."
            raise NotImplementedError(msg)


@dataclasses.dataclass
class RealizationFailure:
    failed_plan_step: PlanStep
