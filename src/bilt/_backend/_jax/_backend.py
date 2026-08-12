import ast
import typing

import jax
import jax.numpy as jnp

from bilt._backend import Backend, BackendCallable, BackendCase
from bilt._backend._callable import CallableType
from bilt._backend._jax._ast import generate_chain_case_ast, generate_function_args_ast, generate_function_return, generate_intermediate_calls_ast
from bilt._backend._jax._ast_transformer import JAXASTTransformer
from bilt._backend._jax._supports import _validate  # pyright: ignore[reportPrivateUsage]
from bilt._backend._jax._tensor_handling import derive_union_domain


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

        function_args = generate_function_args_ast(case)
        function_intermediate_calls = generate_intermediate_calls_ast(case)
        function_return = generate_function_return(case)
        chained_callable_ast = generate_chain_case_ast(f"{case.name}_", function_args, function_intermediate_calls, function_return)

        old_namespace = self.compilation_namespace.copy()
        self.compilation_namespace.update(new_namespace)
        final_callable = self.transpile(chained_callable_ast)
        self.compilation_namespace = old_namespace
        
        return BackendCallable(
            case.name,
            case.parameters,
            case.returns.domain,
            derive_union_domain(case),
            final_callable,
        )
