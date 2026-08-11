import ast
import typing

import jax
import jax.numpy as jnp

from bilt._backend import Backend, BackendCallable, BackendCase
from bilt._backend._callable import CallableType
from bilt._backend._jax._ast import JAXASTTransformer
from bilt._backend._jax._supports import _validate
from bilt.core._tensors._tensor_analysis import DomainedVar, add_dimensions, canonicalised, derive_reshape_slices, identify_extra_dims, is_scalar_reshape, is_unchanged_reshape, make_slice_ast, union
from bilt.symbolic._domains import Domain


class JAXBackend(Backend):
    """JAX backend implementation of the Bilt backend interface."""

    def __init__(self, ast_transformer: ast.NodeTransformer = JAXASTTransformer()) -> None:  # noqa: B008
        self.ast_transformer = ast_transformer
        self.compilation_namespace = {"jnp": jnp, "jax.jit": jax.jit, "__builtins__": __builtins__}  # pyright: ignore[reportUnknownMemberType]

    def transpile(self, function_definition: ast.FunctionDef) -> CallableType:
        """Compile a single transform AST into a callable function."""
        _validate(function_definition)
        return super().transpile(function_definition)

    def chain_case(self, case: BackendCase) -> BackendCallable:
        new_namespace: dict[str, typing.Any] = {}

        function_args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in case.parameters.keys()],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        )

        function_body: list[ast.stmt] = []

        # Extract all available variables in this chain case scope, both from the parameters
        # of chain case, and from the return variables of any intermediates.
        vars: dict[str, DomainedVar] = {}

        for param in case.parameters.values():
            vars[param.name] = param

        for callable in case.intermediates.values():
            var_name = callable.name + "_"
            vars[var_name] = DomainedVar(var_name, callable.return_domain)

        # Assuming for now that we don't need to transpose any domains

        for callable in case.intermediates.values():
            vars_to_use = {
                param.name: vars.get(param.name) or vars[param.name + "_"]
                for param in callable.parameters.values()
            }

            union_input_domain = canonicalised(union(*[vars_to_use[name].domain for name in callable.parameters.keys()]))
            union_callable_domain = callable.union_domain
            extra_dimensions = identify_extra_dims(union_input_domain, union_callable_domain)
            union_domain = canonicalised(union(union_input_domain, union_callable_domain))

            # Now we've traced through extra dimensions within this scope up to this point, update our understanding
            # of the domains of the variable assigned the output of this callable.
            actual_output_domain = union_domain  # Assuming no reduction
            vars[callable.name + "_"] = DomainedVar(callable.name + "_", actual_output_domain)

            call_args: list[ast.expr] = []

            for param in callable.parameters.values():
                source_var = vars_to_use[param.name]
                source_name = source_var.name
                source_domain = source_var.domain
                target_domain = union_domain
                reshape_slices = derive_reshape_slices(source_domain, target_domain)

                if is_unchanged_reshape(reshape_slices) or is_scalar_reshape(reshape_slices):
                    call_args.append(ast.Name(id=source_name, ctx=ast.Load()))
                else:
                    call_args.append(
                        make_slice_ast(source_name, reshape_slices)
                    )

            assignment = ast.Assign(
                targets=[ast.Name(id=f"{callable.name}_", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id=callable.name, ctx=ast.Load()),
                    args=call_args,
                    keywords=[],
                )
            )

            function_body.append(assignment)

        function_body.append(
            ast.Return(value=ast.Name(id=f"{case.returns.name}_", ctx=ast.Load()))
        )

        chained_callable_ast = ast.FunctionDef(
            name=f"{case.name}_",
            args=function_args,
            body=function_body,
            decorator_list=[],
        )

        old_namespace = self.compilation_namespace.copy()
        self.compilation_namespace.update(new_namespace)
        final_callable = self.transpile(chained_callable_ast)
        self.compilation_namespace = old_namespace
        
        return BackendCallable(
            case.name,
            case.parameters,
            case.returns.domain,
            union(*[param.domain for param in case.parameters.values()], case.returns.domain),
            final_callable,
        )
