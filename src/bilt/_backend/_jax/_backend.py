from __future__ import annotations

import ast
import typing

import jax
import jax.numpy as jnp

from bilt._backend import Backend, BackendCallable, BackendCase
from bilt._backend._jax._ast import JAXASTTransformer
from bilt._backend._jax._supports import _validate


class JAXBackend(Backend):
    """JAX backend implementation of the Bilt backend interface."""

    def __init__(self) -> None:
        self.ast_transformer = JAXASTTransformer()
        self.compilation_namespace = {"jnp": jnp, "jax.jit": jax.jit, "__builtins__": __builtins__}

    def transpile(self, function_definition: ast.FunctionDef) -> BackendCallable:
        """Compile a single transform AST into a callable function."""
        _validate(function_definition)
        return super().transpile(function_definition)

    def chain_case(self, case: BackendCase) -> BackendCallable:
        new_namespace: dict[str, typing.Any] = {}
        chained_callable_as_string = f"def {case.name}({', '.join(f'{x}_' for x in case.parameters)}):\n"
        for interim_callable in case.intermediates:
            new_namespace[typing.cast("str", interim_callable.name)] = interim_callable.callable
            chained_callable_as_string += f"    {interim_callable.name}_={interim_callable.name}({', '.join(f'{x}_' for x in interim_callable.parameters)})\n"
        chained_callable_as_string += f"    return {case.return_name}_\n"
        chained_callable_ast = typing.cast("ast.FunctionDef", ast.parse(chained_callable_as_string).body[0])

        old_namespace = self.compilation_namespace.copy()
        self.compilation_namespace.update(new_namespace)
        final_callable = self.transpile(chained_callable_ast)
        self.compilation_namespace = old_namespace

        return final_callable
