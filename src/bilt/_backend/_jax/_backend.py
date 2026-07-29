import ast
import typing

import jax
import jax.numpy as jnp

from bilt._backend import Backend, BackendCallable, BackendCase
from bilt._backend._jax._ast import JAXASTTransformer
from bilt._backend._jax._supports import _validate
from bilt._backend._jax._tensor_handling import align_array

if typing.TYPE_CHECKING:
    from bilt.core._system._analysis import BroadcastPlan


class JAXBackend(Backend):
    """JAX backend implementation of the Bilt backend interface."""

    def __init__(self, ast_transformer: ast.NodeTransformer = JAXASTTransformer()) -> None:  # noqa: B008
        self.ast_transformer = ast_transformer
        self.compilation_namespace = {
            "jax": jax,
            "jax.jit": jax.jit,
            "jnp": jnp,
            "align_array": align_array,
            "__builtins__": __builtins__,
        }

    def apply_broadcast_plan(
        self, function_definition: ast.FunctionDef, broadcast_plan: BroadcastPlan
    ) -> ast.FunctionDef:
        return self._add_tensor_input_alignment_prelude(function_definition, broadcast_plan)

    def transpile(self, function_definition: ast.FunctionDef) -> BackendCallable:
        """Compile a single transform AST into a callable function."""
        _validate(function_definition)
        return super().transpile(function_definition)

    def chain_case(self, case: BackendCase) -> BackendCallable:
        new_namespace: dict[str, typing.Any] = {}
        chained_callable_as_string = f"def {case.name}_({', '.join(f'{x}' for x in case.parameters)}):\n"
        case_parameters = set(case.parameters)
        for interim_callable in case.intermediates:
            new_namespace[interim_callable.name] = interim_callable.callable
            chained_callable_as_string += f"    {interim_callable.name}_={interim_callable.name}({
                ', '.join(
                    parameter if parameter in case_parameters else f'{parameter}_'
                    for parameter in interim_callable.parameters
                )
            })\n"
        chained_callable_as_string += f"    return {case.return_name}_\n"
        chained_callable_ast = typing.cast("ast.FunctionDef", ast.parse(chained_callable_as_string).body[0])

        old_namespace = self.compilation_namespace.copy()
        self.compilation_namespace.update(new_namespace)
        final_callable = self.transpile(chained_callable_ast)
        self.compilation_namespace = old_namespace

        return final_callable

    def _add_tensor_input_alignment_prelude(
        self, func_def: ast.FunctionDef, broadcast_plan: BroadcastPlan
    ) -> ast.FunctionDef:
        args = func_def.args
        param_names = [node.arg for node in (args.posonlyargs + args.args + args.kwonlyargs)]

        prelude_str = ""

        for tensor_plan in broadcast_plan.tensor_plans:
            if tensor_plan.name not in param_names:
                continue

            prelude_str += f"{tensor_plan.name} = align_array({tensor_plan.name}, {len(broadcast_plan.output_domain.dims)}, {tensor_plan.input_to_output_dim_mapping})\n"

        prelude_ast = ast.parse(prelude_str).body

        new_body = func_def.body[:]
        new_body[0:0] = prelude_ast

        return ast.FunctionDef(
            name=func_def.name,
            args=func_def.args,
            body=new_body,
            decorator_list=func_def.decorator_list,
            returns=func_def.returns,
        )
