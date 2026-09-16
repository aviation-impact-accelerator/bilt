import numpy as np
import pytest

from bilt.core._tensors._errors import TensorDomainError
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.symbolic._dimensions import dim
from bilt.symbolic._domains import domain


def test_scalar_tensor() -> None:
    result = tensor(2.0)
    expected = 2.0

    assert result.domain.is_scalar
    assert result.shape == ()
    assert result.ndim == 0
    assert result.values == expected


def test_tensor_over_dimension() -> None:
    time = dim("time")

    result = tensor([1.0, 2.0], over=time)

    assert result.domain == domain(time)
    np.testing.assert_array_equal(result.values, [1.0, 2.0])


def test_tensor_rejects_missing_domain() -> None:
    with pytest.raises(
        TensorDomainError,
        match="Non-scalar values need an explicit domain",
    ):
        tensor([1.0, 2.0])


def test_tensor_rejects_incorrect_rank() -> None:
    time = dim("time")

    with pytest.raises(
        TensorDomainError,
        match="Value rank 2 does not match domain rank 1",
    ):
        Tensor(
            values=np.array([[1.0, 2.0]]),
            domain=domain(time),
        )


def test_tensor_rejects_over_and_domain() -> None:
    time = dim("time")

    with pytest.raises(
        TensorDomainError,
        match="Pass either over= or domain=, not both",
    ):
        tensor(
            [1.0, 2.0],
            over=time,
            domain=domain(time),
        )


def test_tensor_arithmetic_is_not_implemented() -> None:
    time = dim("time")
    left = tensor([1.0, 2.0], over=time)
    right = tensor([3.0, 4.0], over=time)

    with pytest.raises(
        NotImplementedError,
        match="Operations will be dispatched through an active backend",
    ):
        left + right
