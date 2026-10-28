import ast
import functools
import typing

import jax
import jax.numpy as jnp

from bilt._backend._base_backend import ActivatableBackendMixin
from bilt._backend._in_process_python_mixin import InProcessPythonMixin, PythonIntrinsicBindings
from bilt._backend._jax._ast import JAXASTTransformer
from bilt._backend._jax._supports import _validate
from bilt._backend._linear_assembly import (
    LinearLayout,
    assemble_joint,
    assemble_pointwise,
    expand_linear_coefficient,
    linear_basis,
    linear_zero,
    project_joint,
)
from bilt._backend._linear_solve import validate_linear_shapes, validate_linear_solution
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


def _solve_linear(
    matrix: object,
    rhs: object,
    *,
    batch_rank: int | None = None,
    layout: LinearLayout | None = None,
    values: tuple[object, ...] = (),
) -> jax.Array | tuple[object, ...]:
    if layout is not None:
        matrix, rhs = assemble_joint(matrix, rhs, values, layout, xp=jnp)
    elif batch_rank is not None:
        matrix, rhs = assemble_pointwise(matrix, rhs, batch_rank, xp=jnp)
    coefficients = jnp.asarray(matrix)
    rhs_values = jnp.asarray(rhs)
    validate_linear_shapes(coefficients.shape, rhs_values.shape)
    dtype = jnp.result_type(coefficients, rhs_values, jnp.float32)
    solution = jnp.linalg.solve(coefficients.astype(dtype), rhs_values.astype(dtype)[..., None])[..., 0]
    # Python exceptions cannot be raised from traced array predicates. A host
    # callback validates actual values, also when the surrounding case is jitted.
    jax.debug.callback(validate_linear_solution, coefficients, rhs_values, solution, ordered=True)
    return solution if layout is None else project_joint(solution, values, layout, xp=jnp)


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
            linear_solve=_solve_linear,
            linear_basis=functools.partial(linear_basis, xp=jnp),
            linear_expand=functools.partial(expand_linear_coefficient, xp=jnp),
            linear_zero=functools.partial(linear_zero, xp=jnp),
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
        chained_callable_as_string += f"    return {case.return_name}_\n"
        chained_callable_ast = typing.cast(
            "ast.FunctionDef",
            ast.parse(chained_callable_as_string).body[0],
        )

        old_namespace = self.compilation_namespace.copy()
        self.compilation_namespace.update(new_namespace)
        final_callable = self.transpile(chained_callable_ast)
        self.compilation_namespace = old_namespace

        return final_callable


JAX_BACKEND = JAXBackend()
