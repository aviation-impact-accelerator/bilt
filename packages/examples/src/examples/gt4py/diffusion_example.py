"""Run a GT4Py 2D stencil within an ordinary Bilt transform.

Install GT4Py Next separately, then run:

    python -m examples.gt4py.diffusion_example
"""

import numpy as np

from bilt import Implementation, Interface, Tensor, field, system, tensor, transform

from ._diffusion_kernel import x, y


@system
class Diffusion(metaclass=Interface):
    temperature: Tensor = field()
    laplacian: Tensor = field()


@transform
def laplacian(temperature: Tensor) -> Tensor:
    import gt4py.next as gtx
    import numpy as np

    from examples.gt4py._diffusion_kernel import I, J, laplacian_2d, mapping

    inp = mapping.to_runtime_field(temperature)
    out = gtx.as_field([I, J], np.zeros(inp.asnumpy().shape, dtype=np.float64))
    laplacian_2d(inp, out=out[1:-1, 1:-1], offset_provider={})
    return mapping.to_runtime_array(out)


def main() -> None:
    """Check a single hot cell and its nearest neighbours."""
    implementation = Implementation(Diffusion, [laplacian])
    values = np.zeros((7, 5), dtype=np.float64)
    values[3, 2] = 1.0

    # Intentionally supply the tensor with reversed dimensions.
    model = Diffusion(temperature=tensor(values.T, over=(y, x)))
    with implementation:
        result = model.laplacian

    assert result.domain.dims == (y, x)
    np.testing.assert_allclose(result.values[2, 3], -4.0)
    np.testing.assert_allclose(result.values[1, 3], 1.0)
    np.testing.assert_allclose(result.values[2, 2], 1.0)
    print("GT4Py stencil evaluated through a Bilt transform")


if __name__ == "__main__":
    main()
