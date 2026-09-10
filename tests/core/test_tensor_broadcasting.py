from typing import TYPE_CHECKING

import numpy as np
import pytest

from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core import system, transform
from bilt.core._domain_registry import DomainRegistryError
from bilt.core._interface import Interface
from bilt.core._system._field import field
from bilt.core._system._implementation import Implementation
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domains import Domain

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_different_domains_broadcast_over_scalar_transform(
    backend: Backend,
) -> None:
    time = Dim("time")
    node = Dim("node")

    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        b: Tensor = field()
        c: Tensor = field()

    @transform
    def c(a: float, b: float) -> float:
        return a * b

    basic = Implementation(Demo, [c])
    demo = Demo(
        a=tensor([1.0, 2.0, 3.0], over=time),
        b=tensor([10.0, 20.0], over=node),
    )

    with backend, basic:
        result = demo.c

    assert result.domain == Domain((time, node))
    np.testing.assert_allclose(
        result.values,
        np.array(
            [
                [10.0, 20.0],
                [20.0, 40.0],
                [30.0, 60.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_subset_domain_broadcasts_over_missing_dimension(
    backend: Backend,
) -> None:
    time = Dim("time")
    node = Dim("node")

    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        b: Tensor = field()
        c: Tensor = field()

    @transform
    def c(a: float, b: float) -> float:
        return a * b

    basic = Implementation(Demo, [c])
    demo = Demo(
        a=tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            over=(time, node),
        ),
        b=tensor([10.0, 20.0], over=node),
    )

    with backend, basic:
        result = demo.c

    assert result.domain == Domain((time, node))
    np.testing.assert_allclose(
        result.values,
        np.array(
            [
                [10.0, 40.0],
                [30.0, 80.0],
                [50.0, 120.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_partially_overlapping_domains_broadcast_to_union(
    backend: Backend,
) -> None:
    time = Dim("time")
    node = Dim("node")
    scenario = Dim("scenario")

    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        b: Tensor = field()
        c: Tensor = field()

    @transform
    def c(a: float, b: float) -> float:
        return a * b

    basic = Implementation(Demo, [c])
    demo = Demo(
        a=tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            over=(time, node),
        ),
        b=tensor(
            [
                [10.0, 100.0],
                [20.0, 200.0],
                [30.0, 300.0],
            ],
            over=(node, scenario),
        ),
    )

    with backend, basic:
        result = demo.c

    assert result.domain == Domain((time, node, scenario))
    np.testing.assert_allclose(
        result.values,
        np.array(
            [
                [
                    [10.0, 100.0],
                    [40.0, 400.0],
                    [90.0, 900.0],
                ],
                [
                    [40.0, 400.0],
                    [100.0, 1000.0],
                    [180.0, 1800.0],
                ],
            ]
        ),
    )


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_equal_sized_distinct_dimensions_do_not_align(
    backend: Backend,
) -> None:
    time = Dim("time")
    node = Dim("node")

    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        b: Tensor = field()
        c: Tensor = field()

    @transform
    def c(a: float, b: float) -> float:
        return a * b

    basic = Implementation(Demo, [c])
    demo = Demo(
        a=tensor([1.0, 2.0, 3.0], over=time),
        b=tensor([10.0, 20.0, 30.0], over=node),
    )

    with backend, basic:
        result = demo.c

    assert result.domain == Domain((time, node))
    assert result.values.shape == (3, 3)
    np.testing.assert_allclose(
        result.values,
        np.array(
            [
                [10.0, 20.0, 30.0],
                [20.0, 40.0, 60.0],
                [30.0, 60.0, 90.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_broadcast_domain_propagates_through_transform_dag(
    backend: Backend,
) -> None:
    time = Dim("time")
    node = Dim("node")

    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        b: Tensor = field()
        scale: float = field()
        c: Tensor = field()
        d: Tensor = field()

    @transform
    def c(a: float, b: float) -> float:
        return a * b

    @transform
    def d(c: float, scale: float) -> float:
        return c + scale

    basic = Implementation(Demo, [c, d])
    demo = Demo(
        a=tensor([1.0, 2.0, 3.0], over=time),
        b=tensor([10.0, 20.0], over=node),
        scale=1.0,
    )

    with backend, basic:
        result = demo.d

    assert result.domain == Domain((time, node))
    np.testing.assert_allclose(
        result.values,
        np.array(
            [
                [11.0, 21.0],
                [21.0, 41.0],
                [31.0, 61.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_diamond_dag_broadcasts_when_branches_rejoin(
    backend: Backend,
) -> None:
    time = Dim("time")
    node = Dim("node")

    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        b: Tensor = field()
        scale: float = field()
        c: Tensor = field()
        d: Tensor = field()
        e: Tensor = field()

    @transform
    def c(a: float, b: float) -> float:
        return a * b

    @transform
    def d(a: float, scale: float) -> float:
        return a + scale

    @transform
    def e(c: float, d: float) -> float:
        return c + d

    basic = Implementation(Demo, [c, d, e])
    demo = Demo(
        a=tensor([1.0, 2.0, 3.0], over=time),
        b=tensor([10.0, 20.0], over=node),
        scale=1.0,
    )

    with backend, basic:
        result = demo.e

    assert result.domain == Domain((time, node))
    np.testing.assert_allclose(
        result.values,
        np.array(
            [
                [12.0, 22.0],
                [23.0, 43.0],
                [34.0, 64.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_domain_axes_are_aligned_before_scalar_transform(
    backend: Backend,
) -> None:
    time = Dim("time")
    node = Dim("node")

    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        b: Tensor = field()
        c: Tensor = field()

    @transform
    def c(a: float, b: float) -> float:
        return a * b

    basic = Implementation(Demo, [c])
    demo = Demo(
        a=tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            over=(time, node),
        ),
        b=tensor(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
            ],
            over=(node, time),
        ),
    )

    with backend, basic:
        result = demo.c

    assert result.domain == Domain((time, node))
    np.testing.assert_allclose(
        result.values,
        np.array(
            [
                [10.0, 80.0],
                [60.0, 200.0],
                [150.0, 360.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_equal_shaped_reordered_domains_are_aligned(
    backend: Backend,
) -> None:
    time = Dim("time")
    node = Dim("node")

    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        b: Tensor = field()
        c: Tensor = field()

    @transform
    def c(a: float, b: float) -> float:
        return a * b

    basic = Implementation(Demo, [c])
    demo = Demo(
        a=tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            over=(time, node),
        ),
        b=tensor(
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ],
            over=(node, time),
        ),
    )

    with backend, basic:
        result = demo.c

    assert result.domain == Domain((time, node))
    np.testing.assert_allclose(
        result.values,
        np.array(
            [
                [10.0, 60.0],
                [60.0, 160.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_three_overlapping_inputs_broadcast_to_union(
    backend: Backend,
) -> None:
    time = Dim("time")
    node = Dim("node")
    scenario = Dim("scenario")

    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        b: Tensor = field()
        c: Tensor = field()
        result: Tensor = field()

    @transform
    def result(a: float, b: float, c: float) -> float:
        return a + b + c

    basic = Implementation(Demo, [result])
    demo = Demo(
        a=tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            over=(time, node),
        ),
        b=tensor(
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ],
            over=(node, scenario),
        ),
        c=tensor(
            [
                [100.0, 200.0],
                [300.0, 400.0],
            ],
            over=(scenario, time),
        ),
    )

    with backend, basic:
        evaluated = demo.result

    assert evaluated.domain == Domain((time, node, scenario))
    np.testing.assert_allclose(
        evaluated.values,
        np.array(
            [
                [
                    [111.0, 321.0],
                    [132.0, 342.0],
                ],
                [
                    [213.0, 423.0],
                    [234.0, 444.0],
                ],
            ]
        ),
    )


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_output_domain_order_is_independent_of_transform_parameter_order(
    backend: Backend,
) -> None:
    time = Dim("time")
    node = Dim("node")

    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        b: Tensor = field()
        c: Tensor = field()

    @transform
    def c(b: float, a: float) -> float:
        return a * b

    basic = Implementation(Demo, [c])
    demo = Demo(
        a=tensor([1.0, 2.0, 3.0], over=time),
        b=tensor([10.0, 20.0], over=node),
    )

    with backend, basic:
        result = demo.c

    assert result.domain == Domain((time, node))
    np.testing.assert_allclose(
        result.values,
        np.array(
            [
                [10.0, 20.0],
                [20.0, 40.0],
                [30.0, 60.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_separately_constructed_dimensions_with_same_name_align(
    backend: Backend,
) -> None:
    first_time = Dim("time")
    second_time = Dim("time")

    assert first_time == second_time

    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        b: Tensor = field()
        c: Tensor = field()

    @transform
    def c(a: float, b: float) -> float:
        return a + b

    basic = Implementation(Demo, [c])
    demo = Demo(
        a=tensor([1.0, 2.0, 3.0], over=first_time),
        b=tensor([10.0, 20.0, 30.0], over=second_time),
    )

    with backend, basic:
        result = demo.c

    assert result.domain == Domain((first_time,))
    np.testing.assert_allclose(
        result.values,
        np.array([11.0, 22.0, 33.0]),
    )


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_conflicting_sizes_for_same_dimension_raise(
    backend: Backend,
) -> None:
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        b: Tensor = field()
        c: Tensor = field()

    @transform
    def c(a: float, b: float) -> float:
        return a + b

    basic = Implementation(Demo, [c])
    demo = Demo(
        a=tensor([1.0, 2.0, 3.0], over=time),
        b=tensor([10.0, 20.0], over=time),
    )

    with (
        backend,
        basic,
        pytest.raises(
            DomainRegistryError,
            match=r"Dimension time has inconsistent sizes: 3 and 2",
        ),
    ):
        _ = demo.c


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_size_one_does_not_broadcast_over_same_named_dimension(
    backend: Backend,
) -> None:
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        b: Tensor = field()
        c: Tensor = field()

    @transform
    def c(a: float, b: float) -> float:
        return a + b

    basic = Implementation(Demo, [c])
    demo = Demo(
        a=tensor([1.0], over=time),
        b=tensor([10.0, 20.0, 30.0], over=time),
    )

    with (
        backend,
        basic,
        pytest.raises(
            DomainRegistryError,
            match="Dimension time has inconsistent sizes: 1 and 3",
        ),
    ):
        _ = demo.c


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_failed_evaluation_does_not_contaminate_next_evaluation(
    backend: Backend,
) -> None:
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        b: Tensor = field()
        c: Tensor = field()

    @transform
    def c(a: float, b: float) -> float:
        return a + b

    basic = Implementation(Demo, [c])
    invalid = Demo(
        a=tensor([1.0, 2.0, 3.0], over=time),
        b=tensor([10.0, 20.0], over=time),
    )
    valid = Demo(
        a=tensor([1.0, 2.0], over=time),
        b=tensor([10.0, 20.0], over=time),
    )

    with backend, basic:
        with pytest.raises(DomainRegistryError):
            _ = invalid.c

        result = valid.c

    assert result.domain == Domain((time,))
    np.testing.assert_allclose(
        result.values,
        np.array([11.0, 22.0]),
    )


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_scalar_input_broadcasts_over_tensor_domain(
    backend: Backend,
) -> None:
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        scale: float = field()
        c: Tensor = field()

    @transform
    def c(a: float, scale: float) -> float:
        return a * scale

    basic = Implementation(Demo, [c])
    demo = Demo(
        a=tensor([1.0, 2.0, 3.0], over=time),
        scale=10.0,
    )

    with backend, basic:
        result = demo.c

    assert result.domain == Domain((time,))
    np.testing.assert_allclose(
        result.values,
        np.array([10.0, 20.0, 30.0]),
    )


@pytest.mark.parametrize(
    "backend",
    (PYTHON_BACKEND, JAX_BACKEND),
)
def test_each_attribute_evaluation_uses_fresh_domain_registry(
    backend: Backend,
) -> None:
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        scale: float = field()
        c: Tensor = field()

    @transform
    def c(a: float, scale: float) -> float:
        return a * scale

    basic = Implementation(Demo, [c])
    first = Demo(
        a=tensor([1.0, 2.0, 3.0], over=time),
        scale=2.0,
    )
    second = Demo(
        a=tensor([4.0, 5.0], over=time),
        scale=3.0,
    )

    with backend, basic:
        first_result = first.c
        second_result = second.c

    np.testing.assert_allclose(
        first_result.values,
        np.array([2.0, 4.0, 6.0]),
    )
    np.testing.assert_allclose(
        second_result.values,
        np.array([12.0, 15.0]),
    )
