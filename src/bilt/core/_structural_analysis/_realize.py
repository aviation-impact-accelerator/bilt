__all__ = ["RealizationFailure", "realize_plan"]

import ast
import copy
import dataclasses
import typing

from bilt._backend._callable import BackendCallable
from bilt._backend._case import BackendCase
from bilt._backend._errors import BiltBackendError
from bilt.core._equation import BasicEquation
from bilt.core._errors import BiltAnalysisError
from bilt.core._evaluation_context import current_evaluation
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._port import mangle_name
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._tensors._expression_lowering import lower_expression
from bilt.core._tensors._transform_lowering import lower_transform
from bilt.core._to_ast import tree_to_functiondef
from bilt.core._transform import BasicTransform
from bilt.symbolic._expressions import Expr
from bilt.symbolic._isolation._errors import BiltIsolationError
from bilt.symbolic._isolation._isolate import isolate

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend
    from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep
    from bilt.symbolic._domains import Domain


def realize_plan(plan: AnalysisPlan, backend: Backend) -> BackendCase | RealizationFailure:
    """Convert an AnalysisPlan into a BackendCase by compiling each step to a callable."""
    symbol_domains = plan.request.input_domains.copy()
    intermediates = []
    for step in plan.steps:
        result = _operation_to_callable(
            step,
            backend=backend,
            symbol_domains=symbol_domains,
        )
        if isinstance(result, RealizationFailure):
            return result
        intermediates.append(result)

        if result.output_domain is None:
            msg = f"Realized step {step.output!r} has no output domain."
            raise BiltAnalysisError(msg)

        symbol_domains[step.output] = result.output_domain

    output = mangle_name(plan.output)
    return BackendCase(
        name=output,
        parameters=tuple(map(mangle_name, sorted(plan.inputs))),
        intermediates=intermediates,
        return_name=output,
    )


def _operation_to_callable(
    plan_step: PlanStep,
    backend: Backend,
    *,
    symbol_domains: dict[str, Domain],
) -> BackendCallable | RealizationFailure:
    match plan_step.operation:
        case BasicTransform() as transform:
            return _transform_to_callable(
                transform,
                plan_step,
                backend,
                symbol_domains=symbol_domains,
            )

        case BasicEquation() as equation:
            return _equation_to_callable(
                equation,
                plan_step,
                backend,
                symbol_domains=symbol_domains,
            )

        case SubSystemConstituent() as sub_system:
            return _sub_system_to_callable(
                sub_system,
                plan_step,
                backend,
                symbol_domains=symbol_domains,
            )

        case _:
            msg = f"Operation type {type(plan_step.operation)} is not supported for evaluation."
            raise NotImplementedError(msg)


def _transform_to_callable(
    transform: BasicTransform,
    plan_step: PlanStep,
    backend: Backend,
    *,
    symbol_domains: dict[str, Domain],
) -> BackendCallable | RealizationFailure:
    function_ast, output_domain = lower_transform(
        copy.deepcopy(transform._function_ast),  # noqa: SLF001
        function=transform._function,  # noqa: SLF001
        symbol_domains=symbol_domains,
    )

    callable_ = _guarded_transpile(
        function_ast,
        plan_step,
        backend,
    )
    if isinstance(callable_, RealizationFailure):
        return callable_

    return dataclasses.replace(
        callable_,
        output_domain=output_domain,
    )


def _equation_to_callable(
    equation: BasicEquation,
    plan_step: PlanStep,
    backend: Backend,
    *,
    symbol_domains: dict[str, Domain],
) -> BackendCallable | RealizationFailure:
    try:
        target = Expr.symbol(plan_step.output)
        isolated = isolate(
            equation.__bilt_equation_tree__,
            target,
        )
    except BiltIsolationError as error:
        return RealizationFailure(
            failed_plan_step=plan_step,
            cause=error,
        )

    lowered = lower_expression(
        isolated.rhs,
        symbol_domains=symbol_domains,
        canonical_order=current_evaluation().canonical_order,
    )

    function_ast = tree_to_functiondef(
        lowered.tree,
        name=plan_step.output,
        parameter_order=equation.__bilt_parameters__,
    )

    callable_ = _guarded_transpile(
        _NameMangler().visit(function_ast),
        plan_step,
        backend,
    )
    if isinstance(callable_, RealizationFailure):
        return callable_

    return dataclasses.replace(
        callable_,
        output_domain=lowered.domain,
    )


class _NameMangler(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.name = mangle_name(node.name)
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = mangle_name(node.arg)
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = mangle_name(node.id)
        return node


def _sub_system_to_callable(
    sub_system: SubSystemConstituent,
    plan_step: PlanStep,
    backend: Backend,
    *,
    symbol_domains: dict[str, Domain],
) -> BackendCallable | RealizationFailure:
    if plan_step.child_inputs is None:
        msg = f"{sub_system!r} step for {plan_step.output!r} names no route through the child."
        return RealizationFailure(
            failed_plan_step=plan_step,
            cause=BiltAnalysisError(msg),
        )

    child_input_domains = {port: symbol_domains[sub_system.qualify(port)] for port in plan_step.child_inputs}

    child_request = EvaluationRequest(
        output=sub_system.child_port(plan_step.output),
        input_domains=child_input_domains,
    )

    try:
        child_callable = sub_system.implementation.get_evaluation(
            child_request,
            backend,
        )
    except BiltAnalysisError as error:
        return RealizationFailure(
            failed_plan_step=plan_step,
            cause=error,
        )

    return BackendCallable(
        name=mangle_name(plan_step.output),
        parameters=tuple(mangle_name(port) for port in plan_step.consumed),
        callable=child_callable.callable,
        output_domain=child_callable.output_domain,
    )


def _guarded_transpile(
    function_ast: ast.FunctionDef,
    plan_step: PlanStep,
    backend: Backend,
) -> BackendCallable | RealizationFailure:
    try:
        return backend.transpile(function_ast)
    except BiltBackendError as error:
        return RealizationFailure(
            failed_plan_step=plan_step,
            cause=error,
        )


@dataclasses.dataclass
class RealizationFailure:
    """A plan step that could not be compiled to a callable.

    ``cause`` is kept so a search that exhausts every route can report why each was rejected.
    """

    failed_plan_step: PlanStep
    cause: Exception | None = None
