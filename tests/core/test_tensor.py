import numpy as np
import pytest

from bilt.core._tensor._tensor import tensor, BiltTensor
from bilt.core._tensor._error import TensorDomainError
from bilt.symbolic._dim import Dim
from bilt.symbolic._domains import Domain


def test_same_domain_multiplies_elementwise() -> None:
    time = Dim("time")
    domain = Domain((time,))

    lhs = tensor([1.0, 2.0, 3.0], over=time)
    rhs = tensor([10.0, 20.0, 30.0], over=time)

    result = lhs * rhs

    assert result.domain == domain
    np.testing.assert_allclose(result.values, np.array([10.0, 40.0, 90.0]))


def test_scalar_broadcasts_over_domain() -> None:
    time = Dim("time")

    lhs = tensor([1.0, 2.0, 3.0], over=time)

    result = lhs * 10.0

    assert result.domain == Domain((time,))
    np.testing.assert_allclose(result.values, np.array([10.0, 20.0, 30.0]))


def test_explicit_scalar_lift_broadcasts_over_domain() -> None:
    time = Dim("time")

    lhs = tensor([1.0, 2.0, 3.0], over=time)
    rhs = tensor(10.0, over=time)

    result = lhs * rhs

    assert result.domain == Domain((time,))
    np.testing.assert_allclose(result.values, np.array([10.0, 20.0, 30.0]))


def test_different_domains_broadcast_to_combined_domain() -> None:
    time = Dim("time")
    node = Dim("node")

    lhs = tensor([1.0, 2.0, 3.0], over=time)
    rhs = tensor([10.0, 20.0], over=node)

    result = lhs * rhs

    assert result.domain == Domain((time, node))
    assert result.values.shape == (3, 2)
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


def test_higher_domain_multiplied_by_lower_domain() -> None:
    time = Dim("time")
    node = Dim("node")

    lhs = tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        over=(time, node),
    )
    rhs = tensor([10.0, 20.0], over=node)

    result = lhs * rhs

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


def test_domain_order_is_aligned_by_name() -> None:
    time = Dim("time")
    node = Dim("node")

    lhs = tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        over=(time, node),
    )

    rhs = tensor(
        [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ],
        over=(node, time),
    )

    result = lhs * rhs

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


def test_raw_non_scalar_array_needs_explicit_domain() -> None:
    with pytest.raises(TensorDomainError, match="explicit domain"):
        _ = tensor([1.0, 2.0, 3.0])


def test_same_domain_shape_mismatch_raises() -> None:
    time = Dim("time")

    lhs = tensor([1.0, 2.0, 3.0], over=time)
    rhs = tensor([10.0, 20.0], over=time)

    with pytest.raises(TensorDomainError):
        _ = lhs * rhs


def test_rank_must_match_domain_rank() -> None:
    time = Dim("time")
    node = Dim("node")

    with pytest.raises(TensorDomainError):
        _ = BiltTensor(np.array([1.0, 2.0, 3.0]), Domain((time, node)))

def test_bilt_tensor_accepts_matching_rank_domain() -> None:
    time = Dim("time")

    value = BiltTensor(
        np.array([1.0, 2.0, 3.0]),
        Domain((time,)),
    )

    assert value.domain == Domain((time,))
    np.testing.assert_allclose(value.values, np.array([1.0, 2.0, 3.0]))


def test_bilt_tensor_rejects_rank_domain_mismatch() -> None:
    time = Dim("time")
    node = Dim("node")

    with pytest.raises(TensorDomainError):
        _ = BiltTensor(
            np.array([1.0, 2.0, 3.0]),
            Domain((time, node)),
        )


def test_bilt_tensor_scalar_constructor() -> None:
    value = BiltTensor.scalar(10.0)

    assert value.domain == Domain.scalar()
    assert value.values.shape == ()
    assert float(value.values) == 10.0


def test_bilt_tensor_scalar_constructor_rejects_non_scalar() -> None:
    with pytest.raises(TensorDomainError):
        _ = BiltTensor.scalar(np.array([1.0, 2.0, 3.0]))