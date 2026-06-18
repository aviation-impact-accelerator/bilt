"""JAX backend: compile Python AST FunctionDefs to a callable utilising JAX for performance boosts."""

import ast
import copy
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from bilt.jax_backend.validate_ast import _validate

if TYPE_CHECKING:
    import collections.abc

__all__ = ("JaxExecutor", "bilt_asts_to_jax")


class JaxExecutor:
    """Executes a JIT-compiled pipeline, enforcing that only expected inputs are provided at runtime."""

    def __init__(
        self,
        jitted_functions: collections.abc.Mapping[str, collections.abc.Callable],
        order: list[str],
        arg_map: dict[str, list[str]],
        external_inputs: set[str],
    ) -> None:
        """Initialize the JaxExecutor with the JIT-compiled functions, their execution order, argument mapping, and expected external inputs."""
        self.external_inputs = external_inputs
        self._jitted_functions = jitted_functions
        self._order = order
        self._arg_map = arg_map

    def __call__(self, **kwargs: Any) -> Any:  # noqa: ANN401
        """Calls the execution pipeline with the provided keyword arguments, ensuring that only expected inputs are passed."""
        unexpected_additional_args = kwargs.keys() - self.external_inputs
        if unexpected_additional_args:
            msg = f"Unexpected inputs: {unexpected_additional_args}"
            raise ValueError(msg)
        required_missing_args = self.external_inputs - kwargs.keys()
        if required_missing_args:
            msg = f"Missing required inputs: {required_missing_args}"
            raise ValueError(msg)

        intermediates: dict[str, Any] = dict(kwargs)
        for fname in self._order:
            args = [intermediates[a] for a in self._arg_map[fname]]
            intermediates[fname] = self._jitted_functions[fname](*args)
        return intermediates[self._order[-1]]


class _JaxTransformer(ast.NodeTransformer):
    """Transformer to apply our specific optimisations to the AST, such as converting list literals to jnp.array calls."""

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
    """Apply bilt transformations to a ast.FunctionDef and compile it into a callable."""
    _validate(fdef)
    transformed = _JaxTransformer().visit(copy.deepcopy(fdef))
    module = ast.Module(body=[transformed], type_ignores=[])
    ast.fix_missing_locations(module)
    code = compile(module, "<jax_emit>", "exec")
    ns: dict[str, Any] = {"jnp": jnp, "__builtins__": {}}
    exec(code, ns)  # noqa: S102
    return ns[fdef.name]


def extract_external_inputs(function_names: list[str], arg_map: dict[str, list[str]]) -> set[str]:
    """Return args that are not themselves function names — these must be provided externally."""
    all_args = {arg for arg_list in arg_map.values() for arg in arg_list}
    return all_args - set(function_names)


def validate_function_order(ordered_names: list[str], arg_map: dict[str, list[str]], external_inputs: set[str]) -> None:
    """Ensure that each function's arguments are either external inputs or outputs of functions that were called before it."""
    prior: set[str] = set()
    for fname in ordered_names:
        for arg in arg_map[fname]:
            if arg not in external_inputs and arg not in prior:
                msg = f"Argument '{arg}' of '{fname}' is not an expected input or a function that takes place before '{fname}'."
                raise ValueError(msg)
        prior.add(fname)


def bilt_asts_to_jax(fdefs: list[ast.FunctionDef]) -> JaxExecutor:
    """Build a JIT-compiled pipeline from a list of FunctionDefs in execution order.

    If we are given two functions:
    def c(a,b): ...
    def d(c): ...

    We want to produce a callable that takes in [a,b] and returns the output of d.
    The callable will call c(a,b) and then d(c) in order they are passed as an argument, passing the output of c to d.

    Given this is the Jax backend, we will JIT compile each function in the pipeline to the best of our ability.
    """
    function_names = [fdef.name for fdef in fdefs]
    name_to_args = {fdef.name: [a.arg for a in fdef.args.args] for fdef in fdefs}
    external_inputs = extract_external_inputs(function_names, name_to_args)
    validate_function_order(function_names, name_to_args, external_inputs)
    jitted_functions = {fdef.name: jax.jit(transpile(fdef)) for fdef in fdefs}
    return JaxExecutor(jitted_functions, function_names, name_to_args, external_inputs)
