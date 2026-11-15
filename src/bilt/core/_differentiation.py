"""Differentiate realised scalar equation and linear-solve plans before backend execution."""

import ast
import dataclasses
import typing

from bilt._backend import LinearSolveBackend
from bilt._backend._case import BackendCase
from bilt._backend._linear_solve import LINEAR_SOLVE_INTRINSIC_NAME
from bilt.core._port import mangle_name
from bilt.core._structural_analysis._linear_operator_lowering import expression_ast
from bilt.core._system._errors import BiltUnsupportedDerivativeError
from bilt.core._to_ast import tree_to_functiondef
from bilt.symbolic._algebra._expr_ops import is_zero
from bilt.symbolic._differentiation import UnsupportedDerivativeError, jvp_tree
from bilt.symbolic._tree import Tree, int_tree, symbol_tree

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend
    from bilt._backend._callable import BackendCallable
    from bilt.core._structural_analysis._linear_block import LinearBlock


_TANGENT_PREFIX = "__bilt_tangent__"


class _MangleNames(ast.NodeTransformer):
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


def _tangent_seeds(
    inputs: tuple[str, ...], original_inputs: set[str], wrt: str, seen: set[str]
) -> dict[str, Tree]:
    """Seed supplied inputs and refer to already computed tangents."""
    seeds = {}
    for name in inputs:
        compiled_name = mangle_name(name)
        if compiled_name in original_inputs:
            seeds[name] = int_tree(1 if compiled_name == wrt else 0)
        elif compiled_name in seen:
            seeds[name] = symbol_tree(f"{_TANGENT_PREFIX}{compiled_name}")
        else:
            msg = f"Cannot differentiate: input {name!r} has no derivative in the selected plan."
            raise BiltUnsupportedDerivativeError(msg)
    return seeds


def _compile_expression_tangent(
    step: BackendCallable, original_inputs: set[str], wrt: str, seen: set[str], backend: Backend
) -> BackendCallable:
    """Differentiate one acyclic equation, or reject an active opaque operation."""
    if step.symbolic_tree is None:
        return _compile_zero_tangent(step, backend)

    parameters = tuple(name for name in step.symbolic_parameters if mangle_name(name) != step.name)
    seeds = _tangent_seeds(parameters, original_inputs, wrt, seen)
    tangent_parameters = tuple(
        f"{_TANGENT_PREFIX}{mangle_name(name)}"
        for name in parameters
        if mangle_name(name) in seen
    )
    try:
        derivative = jvp_tree(step.symbolic_tree, seeds)
    except UnsupportedDerivativeError as error:
        raise BiltUnsupportedDerivativeError(str(error)) from error

    function = tree_to_functiondef(
        derivative,
        name=f"{_TANGENT_PREFIX}{step.name}",
        parameter_order=(*parameters, *tangent_parameters),
    )
    compiled = backend.transpile(_MangleNames().visit(function))
    return dataclasses.replace(compiled, output_domain=step.output_domain)


def _compile_zero_tangent(step: BackendCallable, backend: Backend) -> BackendCallable:
    function = tree_to_functiondef(int_tree(0), name=f"{_TANGENT_PREFIX}{step.name}", parameter_order=())
    compiled = backend.transpile(function)
    return dataclasses.replace(compiled, output_domain=step.output_domain)


def _function(name: str, parameters: tuple[str, ...], result: ast.expr) -> ast.FunctionDef:
    """Compile an IR operation through the same AST interface as ordinary solves."""
    function = ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=parameter) for parameter in parameters],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Return(value=result)],
        decorator_list=[],
    )
    ast.fix_missing_locations(function)
    return function


def _subscript(value: str, index: int) -> ast.Subscript:
    return ast.Subscript(
        value=ast.Name(id=value, ctx=ast.Load()),
        slice=ast.Constant(index),
        ctx=ast.Load(),
    )


def _tangent_rhs(
    linear: LinearBlock, seeds: dict[str, Tree], solution_name: str
) -> ast.Tuple:
    """Differentiate Ay=b implicitly: A dy = db - dA y."""
    rows: list[ast.expr] = []
    for rhs, coefficients in zip(linear.rhs, linear.coefficients, strict=True):
        tangent_rhs = expression_ast(jvp_tree(rhs, seeds))
        for index, coefficient in enumerate(coefficients):
            tangent_coefficient = jvp_tree(coefficient, seeds)
            if is_zero(tangent_coefficient):
                continue
            correction = ast.BinOp(
                left=expression_ast(tangent_coefficient),
                op=ast.Mult(),
                right=_subscript(solution_name, index),
            )
            tangent_rhs = ast.BinOp(left=tangent_rhs, op=ast.Sub(), right=correction)
        rows.append(tangent_rhs)
    return ast.Tuple(elts=rows, ctx=ast.Load())


def _compile_linear_tangent(
    step: BackendCallable, original_inputs: set[str], wrt: str, seen: set[str], backend: Backend
) -> BackendCallable:
    """Generate a second linear solve from the retained scalar affine IR."""
    linear = step.linear_block
    if linear is None:
        msg = "A tangent linear solve requires scalar affine block metadata."
        raise BiltUnsupportedDerivativeError(msg)
    if not isinstance(backend, LinearSolveBackend):
        msg = "The selected backend cannot compile linear tangent solves."
        raise BiltUnsupportedDerivativeError(msg)

    seeds = _tangent_seeds(tuple(sorted(linear.inputs)), original_inputs, wrt, seen)
    try:
        coefficients = ast.Tuple(
            elts=[
                ast.Tuple(elts=[expression_ast(coefficient) for coefficient in row], ctx=ast.Load())
                for row in linear.coefficients
            ],
            ctx=ast.Load(),
        )
        tangent_rhs = _tangent_rhs(linear, seeds, step.name)
    except UnsupportedDerivativeError as error:
        raise BiltUnsupportedDerivativeError(str(error)) from error

    parameters = (
        *step.parameters,
        *(f"{_TANGENT_PREFIX}{name}" for name in step.parameters if name in seen),
        step.name,
    )
    solve = ast.Call(
        func=ast.Name(id=LINEAR_SOLVE_INTRINSIC_NAME, ctx=ast.Load()),
        args=[coefficients, tangent_rhs],
        keywords=[],
    )
    function = _function(f"{_TANGENT_PREFIX}{step.name}", parameters, solve)
    compiled = backend.transpile_linear_solve(_MangleNames().visit(function))
    return dataclasses.replace(compiled, output_domain=step.output_domain)


def _compile_projection_tangent(step: BackendCallable, backend: Backend) -> BackendCallable:
    """Project an unknown's tangent out of the differentiated solve vector."""
    projection = step.linear_projection
    if projection is None:
        msg = "A linear tangent projection requires projection metadata."
        raise BiltUnsupportedDerivativeError(msg)
    solution_name, index = projection
    tangent_solution = f"{_TANGENT_PREFIX}{solution_name}"
    function = _function(
        f"{_TANGENT_PREFIX}{step.name}",
        (tangent_solution,),
        _subscript(tangent_solution, index),
    )
    return dataclasses.replace(backend.transpile(function), output_domain=step.output_domain)


def differentiate_case(case: BackendCase, *, wrt: str, backend: Backend) -> BackendCase:
    """Interleave scalar derivative calculations with their primal computations."""
    original_inputs = set(case.parameters)
    wrt_name = mangle_name(wrt)
    if wrt_name not in original_inputs:
        msg = f"Derivative input {wrt!r} must be a supplied boundary input."
        raise BiltUnsupportedDerivativeError(msg)

    intermediates: list[BackendCallable] = []
    seen: set[str] = set()
    active: set[str] = {wrt_name}

    for step in case.intermediates:
        intermediates.append(step)
        dependencies = set(step.parameters)
        if step.linear_block is not None:
            tangent = _compile_linear_tangent(step, original_inputs, wrt_name, seen, backend)
        elif step.linear_projection is not None:
            tangent = _compile_projection_tangent(step, backend)
        elif step.symbolic_tree is not None:
            tangent = _compile_expression_tangent(step, original_inputs, wrt_name, seen, backend)
        elif dependencies & active:
            msg = (
                f"Cannot differentiate {step.name!r}: the selected route contains "
                "a transform, subsystem, or coupled solve without an IR differentiation rule."
            )
            raise BiltUnsupportedDerivativeError(msg)
        else:
            tangent = _compile_zero_tangent(step, backend)

        intermediates.append(tangent)
        seen.add(step.name)
        if dependencies & active:
            active.add(step.name)

    return BackendCase(
        name=f"{_TANGENT_PREFIX}{case.name}",
        parameters=case.parameters,
        intermediates=intermediates,
        return_name=f"{_TANGENT_PREFIX}{case.return_name}",
    )
