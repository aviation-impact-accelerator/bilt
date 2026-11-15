"""Differentiate a realised scalar equation plan before backend execution."""

import ast
import dataclasses
import typing

from bilt._backend._case import BackendCase
from bilt.core._port import mangle_name
from bilt.core._system._errors import BiltUnsupportedDerivativeError
from bilt.core._to_ast import tree_to_functiondef
from bilt.symbolic._differentiation import UnsupportedDerivativeError, jvp_tree
from bilt.symbolic._tree import Tree, float_tree, int_tree, symbol_tree

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend
    from bilt._backend._callable import BackendCallable


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


def differentiate_case(
    case: BackendCase, *, wrt: str, backend: Backend, tangent_prefix: str = _TANGENT_PREFIX
) -> BackendCase:
    """Interleave derivative operations with their primal computations."""
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
        tangent_name = f"{tangent_prefix}{step.name}"
        dependencies = set(step.parameters)
        if step.symbolic_tree is None:
            if dependencies & active:
                msg = (
                    f"Cannot differentiate {step.name!r}: the selected route contains "
                    "a transform, subsystem, or coupled solve without an IR differentiation rule."
                )
                raise BiltUnsupportedDerivativeError(msg)
            derivative: Tree = int_tree(0)
            source_parameters: tuple[str, ...] = ()
        else:
            mapping: dict[str, Tree] = {}
            extra_parameters: list[str] = []
            source_parameters = tuple(name for name in step.symbolic_parameters if mangle_name(name) != step.name)
            for parameter in source_parameters:
                compiled_name = mangle_name(parameter)
                if compiled_name in original_inputs:
                    mapping[parameter] = float_tree(1.0 if compiled_name == wrt_name else 0.0)
                elif compiled_name in seen:
                    tangent_parameter = f"{tangent_prefix}{parameter}"
                    mapping[parameter] = symbol_tree(tangent_parameter)
                    extra_parameters.append(tangent_parameter)
                else:
                    msg = (
                        f"Cannot differentiate {step.name!r}: input {parameter!r} "
                        "has no derivative in the selected plan."
                    )
                    raise BiltUnsupportedDerivativeError(msg)
            try:
                derivative = jvp_tree(step.symbolic_tree, mapping)
            except UnsupportedDerivativeError as error:
                raise BiltUnsupportedDerivativeError(str(error)) from error
            source_parameters += tuple(extra_parameters)

        func = tree_to_functiondef(derivative, name=tangent_name, parameter_order=source_parameters)
        compiled = backend.transpile(_MangleNames().visit(func))
        intermediates.append(
            dataclasses.replace(
                compiled,
                output_domain=step.output_domain,
                symbolic_tree=derivative,
                symbolic_parameters=source_parameters,
            )
        )
        seen.add(step.name)
        if dependencies & active:
            active.add(step.name)

    return BackendCase(
        name=f"{tangent_prefix}{case.name}",
        parameters=case.parameters,
        intermediates=intermediates,
        return_name=f"{tangent_prefix}{case.return_name}",
    )
