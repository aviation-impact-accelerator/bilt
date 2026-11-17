"""Regression tests for compilation caching with derived tensor dimensions."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock, patch

from bilt.core._evaluation_context import EvaluationContext, current_evaluation
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._system._implementation import Implementation
from bilt.core._tensors._tensors import tensor
from bilt.symbolic._dimensions import coordinate_axis, dim
from bilt.symbolic._domains import Domain, domain

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


def _implementation() -> Implementation:
    # Exercise the real compilation-cache path without constructing a model.
    implementation = object.__new__(Implementation)
    implementation._evaluation_cache = {}
    return implementation


def _backend() -> Mock:
    backend = Mock()

    def compile_case(_case: object) -> Mock:
        # Compiled operations freeze the axis order after structural planning.
        _ = current_evaluation().canonical_order
        return Mock()

    backend.chain_case.side_effect = compile_case
    return backend


def _case(output_domain: Domain) -> SimpleNamespace:
    operation = SimpleNamespace(name="result", output_domain=output_domain)
    return SimpleNamespace(intermediates=(operation,), return_name="result")


def test_ordinary_plan_cache_reused_for_new_input_extents() -> None:
    time = dim("time")
    request = EvaluationRequest(output="result", input_domains={"input": domain(time)})
    implementation = _implementation()
    backend = _backend()
    analyser = Mock()
    analyser.analyse.return_value = _case(domain(time))

    with patch.object(Implementation, "analyser_for", return_value=analyser):
        for size in (2, 5):
            with EvaluationContext() as evaluation:
                evaluation.domains.bind(tensor([float(i) for i in range(size)], over=time))
                _ = implementation.get_evaluation(request, cast("Backend", backend))

    assert analyser.analyse.call_count == 1
    assert backend.chain_case.call_count == 1


def test_ordinary_plan_cache_separates_coordinate_labels() -> None:
    time = dim("time")
    request = EvaluationRequest(output="result", input_domains={"input": domain(time)})
    implementation = _implementation()
    backend = _backend()
    analyser = Mock()
    analyser.analyse.return_value = _case(domain(time))

    with patch.object(Implementation, "analyser_for", return_value=analyser):
        for years in ((2025, 2026), (2035, 2036)):
            with EvaluationContext() as evaluation:
                evaluation.domains.bind_axis(coordinate_axis(time, years))
                evaluation.domains.bind(tensor([1.0, 2.0], over=time))
                _ = implementation.get_evaluation(request, cast("Backend", backend))

    expected_compilations = 2
    assert analyser.analyse.call_count == expected_compilations
    assert backend.chain_case.call_count == expected_compilations


def test_derived_axes_replanned_across_extent_and_coordinate_changes() -> None:
    historical, projected, combined = dim("historical"), dim("projected"), dim("combined")
    request = EvaluationRequest(
        output="result",
        input_domains={"before": domain(historical), "after": domain(projected)},
    )
    implementation = _implementation()
    backend = _backend()
    analyser = Mock()

    def analyse(_request: EvaluationRequest) -> SimpleNamespace:
        current_evaluation().domains.bind_concat((historical, projected), combined)
        return _case(domain(combined))

    analyser.analyse.side_effect = analyse
    selections = (
        ((2023, 2024), (2025,)),
        ((2023, 2024, 2025), (2026, 2027)),
        ((2033, 2034), (2035,)),
    )

    with patch.object(Implementation, "analyser_for", return_value=analyser):
        for past_years, future_years in selections:
            with EvaluationContext() as evaluation:
                evaluation.domains.bind_axis(coordinate_axis(historical, past_years))
                evaluation.domains.bind_axis(coordinate_axis(projected, future_years))
                evaluation.domains.bind(tensor([1.0] * len(past_years), over=historical))
                evaluation.domains.bind(tensor([2.0] * len(future_years), over=projected))

                _ = implementation.get_evaluation(request, cast("Backend", backend))
                assert evaluation.domains.axis(combined).coordinates == past_years + future_years
                assert evaluation.domains.size(combined) == len(past_years) + len(future_years)

    assert analyser.analyse.call_count == len(selections)
    assert backend.chain_case.call_count == len(selections)
    assert not implementation._evaluation_cache  # noqa: SLF001
