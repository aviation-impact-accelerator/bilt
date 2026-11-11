__all__ = ["BlockRealizationFailure", "BlockRealizationKey", "RealizationFailure", "realize_plan"]

import ast
import copy
import dataclasses
import typing

from bilt._backend._callable import BackendCallable
from bilt._backend._case import BackendCase
from bilt._backend._errors import BiltBackendError
from bilt.core._equation import BasicEquation
from bilt.core._errors import BiltAnalysisError, BiltEquationDomainRequirementError
from bilt.core._evaluation_context import current_evaluation
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._port import mangle_name
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.core._structural_analysis._realize_linear import realize_linear_block
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._tensors._errors import BiltSymbolicLoweringError, BiltTensorDomainError
from bilt.core._tensors._expression_lowering import lower_expression
from bilt.core._tensors._transform_lowering import lower_transform
from bilt.core._to_ast import tree_to_functiondef
from bilt.core._transform import BasicTransform
from bilt.symbolic._domains import Domain
from bilt.symbolic._expressions import Expr
from bilt.symbolic._isolation._errors import BiltIsolationError
from bilt.symbolic._isolation._isolate import isolate

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend._base_backend import Backend
    from bilt.core._structural_analysis._plan import AnalysisPlan, PlanBlock, PlanStep


type BlockRealizationKey = tuple[
    frozenset[tuple[PlanStep, frozenset[str] | None]],
    frozenset[tuple[str, Domain | None]],
]


def realize_plan(
    plan: AnalysisPlan,
    backend: Backend,
    *,
    block_failures: dict[BlockRealizationKey, BlockRealizationFailure] | None = None,
) -> BackendCase | RealizationFailure | BlockRealizationFailure:
    """Realise ordered direct steps and supported linear blocks."""
    symbol_domains = plan.request.input_domains.copy()
    intermediates = []
    for index, block in enumerate(plan.blocks):
        if block.is_cyclic:
            name = f"__bilt_linear_solution_{index}"
            names = {mangle_name(step.output) for step in plan.steps} | set(map(mangle_name, plan.inputs))
            while name in names:
                name += "_"
            realized = _realize_cyclic_with_domains(block, backend, symbol_domains, name, block_failures)
            if isinstance(realized, BlockRealizationFailure):
                return realized
            callables, resolved = realized
            intermediates.extend(callables)
            symbol_domains.update(resolved)
            continue

        step = block.steps[0]
        if isinstance(step.operation, BasicEquation):
            try:
                _check_equation_domains(step, symbol_domains, step.consumed)
            except BiltEquationDomainRequirementError as error:
                return RealizationFailure(failed_plan_step=step, cause=error)
        result = _operation_to_callable(
            step,
            backend=backend,
            symbol_domains=symbol_domains,
        )
        if isinstance(result, RealizationFailure):
            return result
        intermediates.append(result)

        candidate_domain = _output_domain(result, step.output)
        try:
            _check_equation_domains(step, {step.output: candidate_domain}, (step.output,))
        except BiltEquationDomainRequirementError as error:
            return RealizationFailure(failed_plan_step=step, cause=error)
        symbol_domains[step.output] = candidate_domain

    output = mangle_name(plan.output)
    return BackendCase(
        name=output,
        parameters=tuple(map(mangle_name, sorted(plan.inputs))),
        intermediates=intermediates,
        return_name=output,
    )


def _realize_cyclic_with_domains(
    block: PlanBlock,
    backend: Backend,
    known: dict[str, Domain],
    solution_name: str,
    failures: dict[BlockRealizationKey, BlockRealizationFailure] | None,
) -> tuple[tuple[BackendCallable, ...], dict[str, Domain]] | BlockRealizationFailure:
    """Realise a joint block only when its selected equations meet minimum domains."""
    key: BlockRealizationKey = (
        frozenset((step, step.child_inputs) for step in block.steps),
        frozenset((attribute, known.get(attribute)) for attribute in block.inputs),
    )
    if failures is not None and key in failures:
        return failures[key]

    try:
        for step in block.steps:
            _check_equation_domains(step, known, block.inputs)
        callables = realize_linear_block(block, backend, known, solution_name=solution_name)
        resolved = {
            output: _output_domain(callable_, output)
            for output, callable_ in zip(sorted(block.outputs), callables[1:], strict=True)
        }
        for step in block.steps:
            _check_equation_domains(step, known | resolved, block.outputs)
    except (
        BiltLinearBlockError,
        BiltSymbolicLoweringError,
        BiltTensorDomainError,
        BiltEquationDomainRequirementError,
    ) as error:
        failure = BlockRealizationFailure(failed_block=block, cause=error)
        if failures is not None:
            failures[key] = failure
        return failure
    return tuple(callables), resolved


def _check_equation_domains(
    step: PlanStep,
    known: dict[str, Domain],
    names: collections.abc.Iterable[str],
) -> None:
    """Check minimum domains on ports actually used by a selected equation.

    Do not merge these into the system's declarations or exact linear-domain
    hints: a different equation/route may impose different constraints.
    """
    if not isinstance(step.operation, BasicEquation):
        return
    for name in names:
        declaration = step.operation.__bilt_port_declarations__.get(name)
        required = None if declaration is None else declaration.required_domain
        if required is None or name not in known:
            continue
        actual = known[name]
        missing = tuple(dim for dim in required.dims if dim not in actual.dims)
        if missing:
            message = (
                f"Equation {step.operation.__bilt_name__!r} requires port {name!r} "
                f"to include {required}, but inferred {actual}; "
                f"missing dimensions: {', '.join(map(str, missing))}."
            )
            raise BiltEquationDomainRequirementError(message)


def _output_domain(callable_: BackendCallable, output: str) -> Domain:
    if callable_.output_domain is None:
        message = f"Realized step {output!r} has no output domain."
        raise BiltAnalysisError(message)
    return callable_.output_domain


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
    except (BiltIsolationError, BiltSymbolicLoweringError, BiltTensorDomainError) as error:
        return RealizationFailure(
            failed_plan_step=plan_step,
            cause=error,
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

    parent_to_child = {mangle_name(sub_system.qualify(port)): mangle_name(port) for port in child_input_domains}

    def evaluate_child(*args: object, **kwargs: object) -> object:
        return child_callable.callable(
            *args,
            **{parent_to_child[name]: value for name, value in kwargs.items()},
        )

    return BackendCallable(
        name=mangle_name(plan_step.output),
        parameters=tuple(mangle_name(port) for port in plan_step.consumed),
        callable=evaluate_child,
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


@dataclasses.dataclass(frozen=True)
class BlockRealizationFailure:
    """A coupled block rejected during lowering or compilation, before execution.

    This rejects the combined formulation, not any individual equation. The
    caller may memoize it by matched steps, child routes and incoming domains
    within one analysis request. Numerical failures during evaluation do not
    produce this result.
    """

    failed_block: PlanBlock
    cause: Exception
