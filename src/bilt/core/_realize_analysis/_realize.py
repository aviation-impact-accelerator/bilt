__all__ = ["RealizationFailure", "realize_plan"]

import ast
import dataclasses
import inspect
import textwrap
import typing

from bilt._backend._ast import clear_signature_type_hints
from bilt._backend._callable import BackendCallable
from bilt._backend._case import BackendCase
from bilt.core._equation import BasicEquation
from bilt.core._interface import Interface
from bilt.core._tensors._transform_domain_analysis import analyse_domains_in_transform_signature
from bilt.core._transform import BasicTransform
from bilt.symbolic._domain_handling import canonicalised, union
from bilt.symbolic._isolation._errors import IsolationError
from bilt.symbolic._isolation._isolate import equation_to_ast

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend
    from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep
    from bilt.symbolic._domains import CanonicalisedDomain


def realize_plan(plan: AnalysisPlan, backend: Backend) -> BackendCase | RealizationFailure:
    """Convert an AnalysisPlan into a BackendCase by compiling each step to a callable."""
    intermediates: list[BackendCallable] = []
    parameters: dict[str, CanonicalisedDomain] = {}
    intermediate_opts: dict[str, CanonicalisedDomain] = {}

    for step in plan.steps:
        result = _operation_to_callable(step, backend=backend)

        if isinstance(result, RealizationFailure):
            return result

        parameters |= result.expected_parameters

        intermediate_opts[step.output] = result.expected_return_domain
        intermediates.append(result)

    # Assuming no reduction
    output_domain = canonicalised(
        union(
            *[domain.domain for domain in parameters.values()],
            *[domain.domain for domain in intermediate_opts.values()],
        )
    )

    return BackendCase(
        name=plan.output,
        parameters={name: parameters[name] for name in plan.inputs},
        intermediates={intermediate.name: intermediate for intermediate in intermediates},
        return_domain=output_domain,
        return_name=plan.output,
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

            analysis = analyse_domains_in_transform_signature(function_ast)
            clear_signature_type_hints(function_ast)

            transpiled_callable = backend.transpile(function_ast)

            return BackendCallable(
                name=function_ast.name,
                expected_parameters=analysis.parameters,
                expected_return_domain=analysis.return_domain,
                expected_union_domain=analysis.union_domain,
                callable=transpiled_callable,
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
