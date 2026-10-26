"""Lower scalar linear blocks to a solve followed by named scalar projections."""

import ast
import dataclasses
import typing

from bilt._backend import ScalarLinearSolveBackend
from bilt._backend._errors import BiltBackendError
from bilt._backend._linear_solve import LINEAR_SOLVE_INTRINSIC_NAME
from bilt.core._evaluation_context import current_evaluation
from bilt.core._port import mangle_name
from bilt.core._structural_analysis._linear_block import (
    BiltLinearBlockError,
    extract_linear_block,
)
from bilt.core._tensors._expression_lowering import lower_expression
from bilt.core._to_ast import tree_to_functiondef
from bilt.symbolic._domains import Domain
from bilt.symbolic._expressions import Expr

if typing.TYPE_CHECKING:
    from bilt._backend import Backend, BackendCallable
    from bilt.core._structural_analysis._plan import PlanBlock
    from bilt.symbolic._tree import Tree


def realize_linear_block(
    block: PlanBlock,
    backend: Backend,
    symbol_domains: dict[str, Domain],
    *,
    solution_name: str,
) -> tuple[BackendCallable, ...]:
    """Compile one solve and projections, without evaluating any coefficients."""
    if not isinstance(backend, ScalarLinearSolveBackend):
        message = "The selected backend does not support scalar linear blocks."
        raise BiltLinearBlockError(message)
    linear = extract_linear_block(block)
    parameters = tuple(sorted(linear.inputs))

    def scalar(tree: Tree) -> ast.expr:
        lowered = lower_expression(
            Expr(tree), symbol_domains=symbol_domains, canonical_order=current_evaluation().canonical_order
        )
        if not lowered.domain.is_scalar:
            message = f"Scalar linear blocks require scalar coefficients and right-hand sides, got {lowered.domain}."
            raise BiltLinearBlockError(message)
        function = tree_to_functiondef(lowered.tree, name=solution_name, parameter_order=parameters)
        expression = typing.cast("ast.expr", typing.cast("ast.Return", function.body[0]).value)
        for node in ast.walk(expression):
            if isinstance(node, ast.Name):
                node.id = mangle_name(node.id)
        return expression

    coefficients = ast.Tuple(
        elts=[ast.Tuple(elts=[scalar(tree) for tree in row], ctx=ast.Load()) for row in linear.coefficients],
        ctx=ast.Load(),
    )
    rhs = ast.Tuple(elts=[scalar(tree) for tree in linear.rhs], ctx=ast.Load())
    solve = _function(
        solution_name,
        tuple(map(mangle_name, parameters)),
        ast.Call(func=ast.Name(id=LINEAR_SOLVE_INTRINSIC_NAME, ctx=ast.Load()), args=[coefficients, rhs], keywords=[]),
    )
    try:
        callables = [backend.transpile_linear_solve(solve)]
        for index, output in enumerate(linear.unknowns):
            projection = _function(
                mangle_name(output),
                (solution_name,),
                ast.Subscript(
                    value=ast.Name(id=solution_name, ctx=ast.Load()), slice=ast.Constant(index), ctx=ast.Load()
                ),
            )
            callables.append(dataclasses.replace(backend.transpile(projection), output_domain=Domain.scalar()))
    except BiltBackendError as error:
        message = f"The backend could not compile the linear block for {linear.unknowns!r}: {error}"
        raise BiltLinearBlockError(message) from error
    return tuple(callables)


def _function(name: str, parameters: tuple[str, ...], expression: ast.expr) -> ast.FunctionDef:
    function = ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=parameter) for parameter in parameters],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Return(value=expression)],
        decorator_list=[],
    )
    ast.fix_missing_locations(function)
    return function
