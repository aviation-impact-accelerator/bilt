import numpy as np
import pytest

from bilt.core._evaluation_context import EvaluationContext
from bilt.core._system._lowering_and_lifting import (
    generate_lifting_plan,
    lift_output,
)
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domains import domain


def test_lift_output_squeezes_reduced_dimension() -> None:
    location = Dim("location")
    time = Dim("time")

    with EvaluationContext() as evaluation:
        evaluation.domains.bind_all(
            [
                tensor(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                    ],
                    over=(location, time),
                ),
            ]
        )

        lifting_plan = generate_lifting_plan(domain(location))
        result = lift_output(
            np.array(
                [
                    [2.0],
                    [5.0],
                ]
            ),
            lifting_plan,
        )

    assert isinstance(result, Tensor)
    assert result.domain == domain(location)
    np.testing.assert_allclose(
        result.values,
        np.array([2.0, 5.0]),
    )


def test_lift_output_squeezes_all_dimensions_for_scalar() -> None:
    time = Dim("time")

    with EvaluationContext() as evaluation:
        evaluation.domains.bind_all(
            [
                tensor(
                    [1.0, 2.0, 3.0],
                    over=time,
                ),
            ]
        )

        lifting_plan = generate_lifting_plan(domain())
        result = lift_output(
            np.array([2.0]),
            lifting_plan,
        )

    assert not isinstance(result, Tensor)
    assert result == pytest.approx(2.0)
