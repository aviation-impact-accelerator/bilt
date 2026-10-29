import typing

import numpy as np
import pytest

from bilt import Domain, Implementation, Interface, Tensor, dim, domain, equation, field, system, tensor
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._system import _implementation as implementation_module
from bilt.symbolic._dimensions import coordinate_axis

if typing.TYPE_CHECKING:
    from bilt._backend import Backend

TIME = dim("time")
SITE = dim("site")
OverTime = typing.Annotated[Tensor, domain(TIME)]
PerSite = typing.Annotated[Tensor, domain(SITE)]
Shared = typing.Annotated[Tensor, Domain.scalar()]


@pytest.fixture(autouse=True)
def _use_scc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(implementation_module, "CausalizationStructuralAnalyzer", CausalizationStructuralAnalyzer)


@pytest.fixture(params=(PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def backend(request: pytest.FixtureRequest) -> Backend:
    return request.param


@system
class Ports(metaclass=Interface):
    x: Tensor = field()
    y: Tensor = field()
    p: Tensor = field()
    q: Tensor = field()


@equation
def boundary(p: Tensor, inner: Ports) -> bool:
    return inner.x + inner.y == p


@equation
def expose_x(x: Tensor, inner: Ports) -> bool:
    return x == inner.x


@equation
def expose_y(y: Tensor, inner: Ports) -> bool:
    return y == inner.y


def _model(*, hinted: bool = False, depth: int = 0) -> Implementation:
    if hinted:

        @equation
        def aggregate(x: OverTime, y: Shared, q: Tensor) -> bool:
            return 0.5 * x.sum(TIME) - y == q
    else:

        @equation
        def aggregate(x: Tensor, y: Tensor, q: Tensor) -> bool:
            return 0.5 * x.sum(TIME) - y == q

    parent = Implementation(
        Ports, [boundary, expose_x, expose_y], internal_implementations={"inner": Implementation(Ports, [aggregate])}
    )
    for _ in range(depth):

        @equation
        def supply(p: Tensor, inner: Ports) -> bool:
            return inner.p == p

        parent = Implementation(Ports, [supply, expose_x, expose_y], internal_implementations={"inner": parent})
    return parent


@pytest.mark.parametrize("output", ("x", "y"))
@pytest.mark.parametrize("size", (0, 1, 3))
def test_inferred_layout_matches_explicit_layout(backend: Backend, output: str, size: int) -> None:
    inputs: dict[str, object] = {"p": tensor(np.arange(size, dtype=float) + 2, over=TIME), "inner.q": 1.0}
    actual = _model().evaluate(inputs, output, backend)
    expected = _model(hinted=True).evaluate(inputs, output, backend)
    if output == "x":
        assert isinstance(actual, Tensor)
        assert isinstance(expected, Tensor)
        assert actual.domain == expected.domain == domain(TIME)
        np.testing.assert_allclose(actual.values, expected.values, atol=1e-5)
    else:
        assert not isinstance(actual, Tensor)
        assert actual == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("output", ("inner.x", "inner.y"))
def test_subset_domains_follow_known_inputs_and_canonical_order(
    backend: Backend, output: str, *, reverse: bool
) -> None:
    @equation
    def aggregate(x: Tensor, y: Tensor, q: Tensor) -> bool:
        return 0.5 * x.sum(TIME) - y == q

    parent = Implementation(Ports, [boundary], internal_implementations={"inner": Implementation(Ports, [aggregate])})
    axes = (TIME, SITE) if reverse else (SITE, TIME)
    values = np.array([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
    with EvaluationContext() as context:
        for axis in axes:
            context.domains.bind(coordinate_axis(axis, (0, 1, 2) if axis == TIME else (0, 1)))
        result = parent.evaluate(
            {"p": tensor(values, over=(SITE, TIME)), "inner.q": tensor([1.0, 2.0], over=SITE)}, output, backend
        )
    if output == "inner.x":
        assert isinstance(result, Tensor)
        assert result.domain == domain(*axes)
        expected = values - np.array([2.0, 4.0])[:, None]
        np.testing.assert_allclose(result.values, expected.T if reverse else expected, atol=1e-5)
    else:
        assert isinstance(result, Tensor)
        assert result.domain == domain(SITE)
        np.testing.assert_allclose(result.values, [2.0, 4.0], atol=1e-5)


@pytest.mark.parametrize("hint", (None, "x", "y"))
def test_ambiguous_layout_requires_only_a_disambiguating_hint(backend: Backend, hint: str | None) -> None:
    if hint == "x":

        @equation
        def aggregate(x: PerSite, y: Tensor, q: Tensor) -> bool:
            return x.sum(SITE) + y.sum(TIME) == q
    elif hint == "y":

        @equation
        def aggregate(x: Tensor, y: OverTime, q: Tensor) -> bool:
            return x.sum(SITE) + y.sum(TIME) == q
    else:

        @equation
        def aggregate(x: Tensor, y: Tensor, q: Tensor) -> bool:
            return x.sum(SITE) + y.sum(TIME) == q

    parent = Implementation(Ports, [boundary], internal_implementations={"inner": Implementation(Ports, [aggregate])})
    values = np.array([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
    inputs: dict[str, object] = {"p": tensor(values, over=(SITE, TIME)), "inner.q": 1.0}
    if hint is None:
        with pytest.raises(BiltExecutionPlanError, match=r"Ambiguous.*inner.x.*inner.y"):
            parent.evaluate(inputs, "inner.x", backend)
    else:
        result = parent.evaluate(inputs, "inner.x", backend)
        assert isinstance(result, Tensor)
        assert result.domain == (domain(SITE) if hint == "x" else domain(SITE, TIME))
        expected = np.array([15.0, 19.0]) if hint == "x" else values - np.array([-14.0, -11.0, -8.0])[None, :]
        np.testing.assert_allclose(result.values, expected, atol=1e-5)


def test_conflicting_hint_is_rejected(backend: Backend) -> None:
    @equation
    def aggregate(x: Shared, y: Tensor, q: Tensor) -> bool:
        return 0.5 * x.sum(TIME) - y == q

    parent = Implementation(Ports, [boundary], internal_implementations={"inner": Implementation(Ports, [aggregate])})
    with pytest.raises(BiltExecutionPlanError, match="No consistent"):
        parent.evaluate({"p": tensor([2.0, 4.0], over=TIME), "inner.q": 1.0}, "inner.x", backend)


def test_reduction_cannot_invent_an_axis(backend: Backend) -> None:
    with pytest.raises(BiltExecutionPlanError, match="block inputs"):
        _model().evaluate({"p": 2.0, "inner.q": 1.0}, "x", backend)


def test_cached_requests_can_infer_different_layouts(backend: Backend) -> None:
    parent = _model()
    scalar = parent.evaluate({"p": tensor([2.0, 4.0, 6.0], over=TIME), "inner.q": 1.0}, "y", backend)
    assert scalar == pytest.approx(2.0, abs=1e-5)
    subset = parent.evaluate(
        {"p": tensor([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]], over=(SITE, TIME)), "inner.q": tensor([1.0, 2.0], over=SITE)},
        "y",
        backend,
    )
    assert isinstance(subset, Tensor)
    assert subset.domain == domain(SITE)
    np.testing.assert_allclose(subset.values, [2.0, 4.0], atol=1e-5)


@pytest.mark.parametrize("depth", (1, 2))
@pytest.mark.parametrize("output", ("x", "y"))
def test_nested_aliases_retain_inference_evidence(backend: Backend, depth: int, output: str) -> None:
    parent = _model(depth=depth)
    result = parent.evaluate(
        {"p": tensor([2.0, 4.0, 6.0], over=TIME), ".".join(("inner",) * (depth + 1) + ("q",)): 1.0}, output, backend
    )
    if output == "x":
        assert isinstance(result, Tensor)
        assert result.domain == domain(TIME)
        np.testing.assert_allclose(result.values, [0.0, 2.0, 4.0], atol=1e-5)
    else:
        assert result == pytest.approx(2.0, abs=1e-5)


@pytest.mark.parametrize("selection", ("isel", "sel"))
def test_selection_infers_both_tensor_unknowns(backend: Backend, selection: str) -> None:
    @equation
    def selected(x: Tensor, y: Tensor, q: Tensor) -> bool:
        return x - 0.25 * getattr(y, selection)(TIME, 0 if selection == "isel" else "first") == q

    parent = Implementation(Ports, [boundary], internal_implementations={"inner": Implementation(Ports, [selected])})
    with EvaluationContext() as context:
        context.domains.bind(coordinate_axis(TIME, ("first", "second")))
        result = parent.evaluate(
            {"p": tensor([2.0, 4.0], over=TIME), "inner.q": tensor([0.0, 1.0], over=TIME)}, "inner.x", backend
        )
    assert isinstance(result, Tensor)
    assert result.domain == domain(TIME)
    np.testing.assert_allclose(result.values, [0.4, 1.4], atol=1e-5)


def test_reused_child_implementations_infer_instance_specific_domains(backend: Backend) -> None:
    @system
    class Parent(metaclass=Interface):
        answer: Tensor = field()

    @equation
    def combine(answer: Tensor, left: Ports, right: Ports) -> bool:
        return answer == left.y + right.y

    child = _model()
    parent = Implementation(Parent, [combine], internal_implementations={"left": child, "right": child})
    result = parent.evaluate(
        {
            "left.p": tensor([2.0, 4.0, 6.0], over=TIME),
            "left.inner.q": 1.0,
            "right.p": tensor([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]], over=(SITE, TIME)),
            "right.inner.q": tensor([1.0, 2.0], over=SITE),
        },
        "answer",
        backend,
    )
    assert isinstance(result, Tensor)
    assert result.domain == domain(SITE)
    np.testing.assert_allclose(result.values, [4.0, 6.0], atol=1e-5)
