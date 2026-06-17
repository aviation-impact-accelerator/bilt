"""JAX backend: compile Python AST FunctionDefs to a callable utilising JAX for performance boosts."""

import ast
import copy
import typing
from typing import Any

import jax
import jax.numpy as jnp

from bilt.jax_backend.supported_node_check import _validate

if typing.TYPE_CHECKING:
    import collections.abc

__all__ = ("JaxExecutor", "bilt_asts_to_jax")


class JaxExecutor:
    """Wrapper for the callable that executes the plan provided by structural analysis, enforcing that only expected inputs are provided at runtime."""

    def __init__(self, fn: collections.abc.Callable, external_inputs: set[str]) -> None:
        """Initialize the JaxExecutor with the callable and the set of expected external inputs."""
        self.external_inputs = external_inputs
        self._fn = fn

    def __call__(self, **kwargs: Any) -> Any:  # noqa: ANN401
        """Calls the executor function with the provided keyword arguments, ensuring that only expected inputs are passed."""
        unexpected = kwargs.keys() - self.external_inputs
        if unexpected:
            msg = f"Unexpected inputs: {unexpected}"
            raise ValueError(msg)
        return self._fn(**kwargs)


class _JaxTransformer(ast.NodeTransformer):
    def visit_List(self, node: ast.List) -> ast.Call:
        self.generic_visit(node)
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="jnp", ctx=ast.Load()),
                attr="array",
                ctx=ast.Load(),
            ),
            args=[node],
            keywords=[],
        )


def transpile(fdef: ast.FunctionDef) -> collections.abc.Callable:
    _validate(fdef)
    transformed = _JaxTransformer().visit(copy.deepcopy(fdef))
    module = ast.Module(body=[transformed], type_ignores=[])
    ast.fix_missing_locations(module)
    code = compile(module, "<jax_emit>", "exec")
    ns: dict[str, Any] = {"jnp": jnp, "__builtins__": {}}
    exec(code, ns)  # noqa: S102
    return ns[fdef.name]


def bilt_asts_to_jax(
    fdefs: list[ast.FunctionDef],
) -> JaxExecutor:
    """Build a JIT-compiled pipeline from a list of FunctionDefs in execution order.

    If we are given two functions:
    def c(a,b): ...
    def d(c): ...

    We want to produce a callable that takes in [a,b] and returns the output of d.
    The callable will call c(a,b) and then d(c) in order they are passed as an argument, passing the output of c to d.

    Given this is the Jax backend, we will JIT compile each function in the pipeline to the best of our ability.
    """
    order = [fdef.name for fdef in fdefs]
    arg_map = {fdef.name: [a.arg for a in fdef.args.args] for fdef in fdefs}

    external_inputs: set[str] = set()
    for args in arg_map.values():
        for arg in args:
            if arg not in order:
                external_inputs.add(arg)

    prior: set[str] = set()
    for fname in order:
        for arg in arg_map[fname]:
            if arg not in external_inputs and arg not in prior:
                msg = f"Argument '{arg}' of '{fname}' is not satisfied by any external input or prior function output."
                raise ValueError(msg)
        prior.add(fname)

    jitted = {fdef.name: jax.jit(transpile(fdef)) for fdef in fdefs}

    def pipeline(**kwargs: typing.Any) -> typing.Any:  # noqa: ANN401
        intermediates: dict[str, Any] = dict(kwargs)
        for fname in order:
            args = [intermediates[a] for a in arg_map[fname]]
            intermediates[fname] = jitted[fname](*args)
        return intermediates[order[-1]]

    return JaxExecutor(pipeline, external_inputs)
