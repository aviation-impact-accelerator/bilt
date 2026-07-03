from __future__ import annotations

import ast
import inspect
import typing

import jax
import jax.numpy as jnp

from bilt._backend import Backend, BackendCase
from bilt._backend._jax._ast import JaxASTTransformer
from bilt._backend._jax._supports import _validate

if typing.TYPE_CHECKING:
    import collections.abc


class JaxBackend(Backend):
    """JAX backend implementation of the Bilt backend interface."""

    def __init__(self) -> None:
        self.ast_transformer = JaxASTTransformer()
        self.compilation_namespace = {"jnp": jnp, "jax.jit": jax.jit, "__builtins__": __builtins__}

    def transpile(self, function_definition: ast.FunctionDef) -> collections.abc.Callable:
        """Compile a single transform AST into a callable function."""
        _validate(function_definition)
        return super().transpile(function_definition)

    def chain_case(self, case: BackendCase) -> collections.abc.Callable:
        new_namespace: dict[str, typing.Any] = {}
        args = "" if not case.parameters else "_, ".join(case.parameters) + "_"
        chained_callable_as_string = f"def {case.name}({args}):\n"
        for interim_callable in case.interim:
            if not hasattr(interim_callable, "__name__"):
                msg = f"Interim callable {interim_callable} does not have a __name__ attribute."
                raise ValueError(msg)
            new_namespace[typing.cast("str", interim_callable.__name__)] = interim_callable
            args = (
                ""
                if not inspect.signature(interim_callable).parameters
                else "_, ".join(inspect.signature(interim_callable).parameters) + "_"
            )
            chained_callable_as_string += f"    {interim_callable.__name__}_={interim_callable.__name__}({args})\n"
        chained_callable_as_string += f"    return {case.return_name}_\n"
        chained_callable_ast = typing.cast("ast.FunctionDef", ast.parse(chained_callable_as_string).body[0])

        old_namespace = self.compilation_namespace.copy()
        self.compilation_namespace.update(new_namespace)
        final_callable = self.transpile(chained_callable_ast)
        self.compilation_namespace = old_namespace

        return final_callable
