"""Regression checks for pointwise and shared linear layouts without Field aliases."""

import pytest
from linear_models import LinearCase, LinearTrial, linear_cases


@pytest.mark.parametrize("case", linear_cases(extended=False), ids=lambda case: case.label)
def test_polymorphic_linear_layouts(case: LinearCase) -> None:
    trial = LinearTrial.prepare(case, "evaluation")
    try:
        trial.analyse()
        assert trial.check()["status"] == "success"
    finally:
        trial.close()
