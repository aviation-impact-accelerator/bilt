from typing import TYPE_CHECKING

import numpy as np
import pytest

from bilt._backend._python._backend import PythonBackend
from bilt.core import system, transform
from bilt.core._system._field import field
from bilt.core._system._implementation import Implementation
from bilt.core._system._interface import Interface
from bilt.core._tensor._tensor import BiltTensor, TensorDomainError, tensor
from bilt.symbolic._dim import Dim
from bilt.symbolic._domains import Domain

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


@pytest.mark.parametrize(
    ("backend"),
    (PythonBackend(),),
)
def test_different_domains_broadcast_over_scalar_transform(
    backend: Backend,
) -> None:
    time = Dim("time")
    node = Dim("node")

    @system
    class Demo(metaclass=Interface):
        a: BiltTensor = field()
        b: BiltTensor = field()
        c: BiltTensor = field()

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
    ("backend"),
    (PythonBackend(),),
)
def test_broadcast_domain_propagates_through_transform_dag(
    backend: Backend,
) -> None:
    time = Dim("time")
    node = Dim("node")

    @system
    class Demo(metaclass=Interface):
        a: BiltTensor = field()
        b: BiltTensor = field()
        scale: float = field()
        c: BiltTensor = field()
        d: BiltTensor = field()

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
    ("backend"),
    (PythonBackend(),),
)
def test_domain_axes_are_aligned_before_scalar_transform(
    backend: Backend,
) -> None:
    time = Dim("time")
    node = Dim("node")

    @system
    class Demo(metaclass=Interface):
        a: BiltTensor = field()
        b: BiltTensor = field()
        c: BiltTensor = field()

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
    ("backend"),
    (PythonBackend(),),
)
def test_conflicting_sizes_for_same_dimension_raise(
    backend: Backend,
) -> None:
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        a: BiltTensor = field()
        b: BiltTensor = field()
        c: BiltTensor = field()

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
            TensorDomainError,
            match="expected 3",
        ),
    ):
        _ = demo.c


@pytest.mark.parametrize(
    ("backend"),
    (PythonBackend(),),
)
def test_scalar_input_broadcasts_over_tensor_domain(
    backend: Backend,
) -> None:
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        a: BiltTensor = field()
        scale: float = field()
        c: BiltTensor = field()

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
    ("backend"),
    (PythonBackend(),),
)
def test_each_attribute_evaluation_uses_fresh_domain_registry(
    backend: Backend,
) -> None:
    time = Dim("time")

    @system
    class Demo(metaclass=Interface):
        a: BiltTensor = field()
        scale: float = field()
        c: BiltTensor = field()

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
