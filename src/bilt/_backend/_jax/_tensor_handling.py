import jax
import jax.numpy as jnp


def align_array(
    input_array: jax.Array, num_output_dimensions: int, input_to_output_dim_mapping: tuple[int, ...]
) -> jax.Array:
    """Transposes and inserts dimensions in input array to match `num_output_dimensions` and `input_to_output_dim_mapping`.

    jax.lax.broadcast_in_dim doesn't transpose square matrices for some bizarre reason, so this makes the transposition explicit and fixes that bug.

    This function is jittable.
    """
    extra_dims_required = num_output_dimensions - input_array.ndim
    padded_array = input_array[(...,) + (jnp.newaxis,) * extra_dims_required]

    output_to_input_dim_mapping: list[int] = []
    next_extra_dim = input_array.ndim
    for output_dim in range(num_output_dimensions):
        if output_dim in input_to_output_dim_mapping:
            input_dim = input_to_output_dim_mapping.index(output_dim)
            output_to_input_dim_mapping.append(input_dim)
        else:
            output_to_input_dim_mapping.append(next_extra_dim)
            next_extra_dim += 1

    return jax.lax.transpose(padded_array, output_to_input_dim_mapping)
