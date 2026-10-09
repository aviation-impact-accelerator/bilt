import numpy as np
import pytest

from bilt.core._tensors._tensors import Tensor, tensor
from bilt.errors import BiltDimensionalityError, BiltTensorDomainError, BiltUnitError
from bilt.symbolic._dimensions import dim
from bilt.symbolic._domains import domain
from bilt.units import (
    absolute_celsius,
    absolute_kelvin,
    absolute_year,
    gram,
    kelvin,
    kilogram,
    meter,
    second,
)


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
        BiltTensorDomainError,
        match="Non-scalar values need an explicit domain",
    ):
        tensor([1.0, 2.0])


def test_tensor_rejects_incorrect_rank() -> None:
    time = dim("time")

    with pytest.raises(
        BiltTensorDomainError,
        match="Value rank 2 does not match domain rank 1",
    ):
        Tensor(
            values=np.array([[1.0, 2.0]]),
            domain=domain(time),
        )


def test_tensor_rejects_over_and_domain() -> None:
    time = dim("time")

    with pytest.raises(
        BiltTensorDomainError,
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


def test_a_tensor_has_no_unit_by_default() -> None:
    assert tensor([1.0, 2.0], over=dim("time")).unit is None


def test_a_tensor_takes_a_unit_at_construction() -> None:
    result = tensor([1.0, 2.0], over=dim("time"), unit=gram)

    assert result.unit is gram
    np.testing.assert_array_equal(result.values, [1.0, 2.0])


def test_a_scalar_tensor_keeps_its_unit() -> None:
    result = tensor(3.0, unit=gram)

    assert result.domain.is_scalar
    assert result.unit is gram


def test_multiplying_an_unlabelled_tensor_by_a_unit_labels_it() -> None:
    time = dim("time")

    result = tensor([1.0, 2.0], over=time) * gram

    assert result.unit is gram
    assert result.domain == domain(time)
    np.testing.assert_array_equal(result.values, [1.0, 2.0])


def test_a_unit_may_not_premultiply_an_unlabelled_tensor() -> None:
    with pytest.raises(TypeError):
        gram * tensor([1.0, 2.0], over=dim("time"))


def test_an_unlabelled_tensor_may_not_be_divided_by_a_unit() -> None:
    with pytest.raises(TypeError):
        tensor([1.0, 2.0], over=dim("time")) / gram


def test_multiplying_a_labelled_tensor_by_a_unit_composes_the_units() -> None:
    labelled = tensor([1.0, 2.0], over=dim("time"), unit=gram)

    result = labelled * meter

    assert result.unit == gram * meter
    np.testing.assert_array_equal(result.values, [1.0, 2.0])


def test_a_unit_premultiplying_a_labelled_tensor_composes_the_units() -> None:
    labelled = tensor([1.0, 2.0], over=dim("time"), unit=gram)

    result = meter * labelled

    assert result.unit == meter * gram
    np.testing.assert_array_equal(result.values, [1.0, 2.0])


def test_dividing_a_labelled_tensor_by_a_unit_composes_the_units() -> None:
    labelled = tensor([1.0, 2.0], over=dim("time"), unit=gram)

    result = labelled / second

    assert result.unit == gram / second
    np.testing.assert_array_equal(result.values, [1.0, 2.0])


def test_a_unit_may_not_be_divided_by_a_tensor() -> None:
    labelled = tensor([1.0, 2.0], over=dim("time"), unit=gram)

    with pytest.raises(NotImplementedError, match="Operations will be dispatched through an active backend"):
        gram / labelled


def test_labelling_a_tensor_with_an_absolute_unit_composes_as_a_scalar_does() -> None:
    labelled = tensor([1.0, 2.0], over=dim("time"), unit=gram)

    result = labelled * absolute_celsius

    assert result.unit == gram * absolute_celsius


@pytest.mark.parametrize(
    "operand",
    (
        pytest.param(2.0, id="number"),
        pytest.param(tensor([3.0, 4.0], over=dim("other")), id="tensor"),
    ),
)
def test_tensor_multiplication_outside_the_unit_system_still_raises(operand: object) -> None:
    labelled = tensor([1.0, 2.0], over=dim("time"), unit=gram)

    with pytest.raises(NotImplementedError, match="Operations will be dispatched through an active backend"):
        labelled * operand


def test_negating_a_labelled_tensor_still_raises() -> None:
    labelled = tensor([1.0, 2.0], over=dim("time"), unit=gram)

    with pytest.raises(NotImplementedError, match="Operations will be dispatched through an active backend"):
        _ = -labelled


def test_converting_a_tensor_rescales_its_values() -> None:
    time = dim("time")
    labelled = tensor([1000.0, 2000.0], over=time, unit=gram)

    result = labelled.convert_to(kilogram)

    assert result.unit is kilogram
    assert result.domain == domain(time)
    np.testing.assert_allclose(result.values, [1.0, 2.0])


def test_converting_an_absolute_tensor_applies_the_offset() -> None:
    labelled = tensor([0.0, 100.0], over=dim("time"), unit=absolute_celsius)

    result = labelled.convert_to(absolute_kelvin)

    assert result.unit is absolute_kelvin
    np.testing.assert_allclose(result.values, [273.15, 373.15])


def test_converting_a_tensor_to_its_own_absolute_unit_leaves_it_alone() -> None:
    labelled = tensor([2020.0, 2021.0], over=dim("time"), unit=absolute_year)

    result = labelled.convert_to(absolute_year)

    assert result.unit is absolute_year
    np.testing.assert_array_equal(result.values, [2020.0, 2021.0])


def test_converting_a_tensor_to_a_wrong_dimensionality_raises() -> None:
    labelled = tensor([1.0, 2.0], over=dim("time"), unit=gram)

    with pytest.raises(BiltDimensionalityError):
        labelled.convert_to(meter)


def test_converting_a_tensor_between_absolute_and_relative_raises() -> None:
    labelled = tensor([1.0, 2.0], over=dim("time"), unit=absolute_celsius)

    with pytest.raises(BiltUnitError):
        labelled.convert_to(kelvin)


def test_converting_an_unlabelled_tensor_raises_and_names_the_label() -> None:
    unlabelled = tensor([1.0, 2.0], over=dim("time"))

    with pytest.raises(BiltUnitError, match=r"tensor\(\.\.\., unit=\.\.\.\)"):
        unlabelled.convert_to(kilogram)
