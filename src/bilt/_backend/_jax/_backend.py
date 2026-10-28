import ast
import typing

import jax
import jax.numpy as jnp

from bilt._backend._base_backend import ActivatableBackendMixin
from bilt._backend._in_process_python_mixin import InProcessPythonMixin, PythonIntrinsicBindings
from bilt._backend._jax._ast import JAXASTTransformer
from bilt._backend._jax._supports import _validate
from bilt._backend._ndarray_alignment_mixin import NDArrayAlignmentMixin

if typing.TYPE_CHECKING:
    from bilt._backend import BackendCallable, BackendCase


def _isel(
    value: jax.Array,
    index: int | jax.Array,
    *,
    axis: int,
) -> jax.Array:
    return jnp.take(value, jnp.asarray(index)[None], axis=axis)


class JAXBackend(InProcessPythonMixin, ActivatableBackendMixin, NDArrayAlignmentMixin):
    """JAX backend implementation of the Bilt backend interface."""

    def __init__(self) -> None:
        self.ast_transformer = JAXASTTransformer()
        self.compilation_namespace = {
            "jnp": jnp,
            "jax.jit": jax.jit,
        }
        self.intrinsic_bindings = PythonIntrinsicBindings(
            isel=_isel,
            mean=jnp.mean,
            sum=jnp.sum,
            min=jnp.min,
            max=jnp.max,
            prod=jnp.prod,
        )

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
            chained_callable_as_string += f"""    {interim_callable.name}_={interim_callable.name}({
                ", ".join(
                    parameter if parameter in case_parameters else f"{parameter}_"
                    for parameter in interim_callable.parameters
                )
            })\n"""
        if case.expose_intermediates:
            returns = ", ".join(
                output.backend_name if output.backend_name in case_parameters else f"{output.backend_name}_"
                for output in case.outputs
            )
            chained_callable_as_string += f"    return ({returns},)\n" if returns else "    return ()\n"
        else:
            return_name = case.return_name if case.return_name in case_parameters else f"{case.return_name}_"
            chained_callable_as_string += f"    return {return_name}\n"
        chained_callable_ast = typing.cast(
            "ast.FunctionDef",
            ast.parse(chained_callable_as_string).body[0],
        )

        old_namespace = self.compilation_namespace.copy()
        self.compilation_namespace.update(new_namespace)
        final_callable = self.transpile(chained_callable_ast)
        self.compilation_namespace = old_namespace

        final_callable.outputs = case.outputs if case.expose_intermediates else None
        return final_callable


JAX_BACKEND = JAXBackend()
