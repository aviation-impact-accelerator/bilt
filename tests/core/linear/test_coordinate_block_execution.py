"""End-to-end execution of coordinate-matched mixed scalar and tensor blocks."""

import typing

import numpy as np
import pytest

from bilt import Implementation, Interface, Tensor, dim, domain, equation, field, system, tensor

if typing.TYPE_CHECKING:
    from bilt._backend import Backend

WS = dim("wing_source")
TS = dim("tail_source")
WT = dim("wing_target")
TT = dim("tail_target")

WingWing = typing.Annotated[Tensor, domain(WT, WS)]
WingTail = typing.Annotated[Tensor, domain(WT, TS)]
TailWing = typing.Annotated[Tensor, domain(TT, WS)]
TailTail = typing.Annotated[Tensor, domain(TT, TS)]
WingGamma = typing.Annotated[Tensor, domain(WS)]
TailGamma = typing.Annotated[Tensor, domain(TS)]
WingRHS = typing.Annotated[Tensor, domain(WT)]
TailRHS = typing.Annotated[Tensor, domain(TT)]


@system
class TrimBlock(metaclass=Interface):
    """Two independent wing and tail unknown tensors, plus two scalars."""

    ww: WingWing = field()
    wt: WingTail = field()
    tw: TailWing = field()
    tt: TailTail = field()
    wing_width: WingGamma = field()
    tail_width: TailGamma = field()
    wing_arm: WingGamma = field()
    tail_arm: TailGamma = field()
    wing_rhs: WingRHS = field()
    tail_rhs: TailRHS = field()
    total: float = field()
    moment: float = field()
    wing: WingGamma = field()
    tail: TailGamma = field()
    alpha: float = field()
    elevator: float = field()


@equation
def wing_boundary(  # noqa: PLR0913, PLR0917 - physical named ports.
    ww: WingWing, wt: WingTail, wing: WingGamma, tail: TailGamma, alpha: float, wing_rhs: WingRHS
) -> bool:
    """Two aerodynamic boundary rows on the wing."""
    return (ww * wing).sum(WS) + (wt * tail).sum(TS) + alpha == wing_rhs


@equation
def tail_boundary(  # noqa: PLR0913, PLR0917 - physical named ports.
    tw: TailWing,
    tt: TailTail,
    wing: WingGamma,
    tail: TailGamma,
    alpha: float,
    elevator: float,
    tail_rhs: TailRHS,
) -> bool:
    """Two aerodynamic boundary rows on the tail."""
    return (tw * wing).sum(WS) + (tt * tail).sum(TS) + alpha + elevator == tail_rhs


@equation
def total_balance(
    wing: WingGamma,
    tail: TailGamma,
    wing_width: WingGamma,
    tail_width: TailGamma,
    total: float,
) -> bool:
    """Scalar sum of wing and tail sectional forces."""
    return (wing * wing_width).sum(WS) + (tail * tail_width).sum(TS) == total


@equation
def moment_balance(  # noqa: PLR0913, PLR0917 - physical named ports.
    wing: WingGamma,
    tail: TailGamma,
    wing_width: WingGamma,
    tail_width: TailGamma,
    wing_arm: WingGamma,
    tail_arm: TailGamma,
    moment: float,
) -> bool:
    """Scalar moment around the reference centre of gravity."""
    return (wing * wing_width * wing_arm).sum(WS) + (tail * tail_width * tail_arm).sum(TS) == moment


@pytest.mark.parametrize("output", ("wing", "tail", "alpha", "elevator"))
def test_two_source_tensors_and_two_global_scalars(backend: Backend, output: str) -> None:
    """A nominal six-by-six solve must include all four equations in one block."""
    a = np.array(
        [
            [-0.35, -0.07, -0.015, -0.005],
            [-0.065, -0.34, -0.008, -0.014],
            [-0.06, -0.045, -0.59, -0.075],
            [-0.05, -0.065, -0.065, -0.61],
        ]
    )
    widths = np.array([4.1, 3.9, 2.1, 1.9])
    arms = np.array([0.5, 0.55, -3.9, -4.1])
    expected = np.array([7.7, 6.1, 2.3, 1.4, 0.04, -0.01])
    coefficient = np.zeros((6, 6))
    coefficient[:4, :4] = a
    coefficient[:4, 4] = 1.0
    coefficient[2:4, 5] = 1.0
    coefficient[4, :4] = widths
    coefficient[5, :4] = widths * arms
    right = coefficient @ expected
    with backend, Implementation(TrimBlock, [wing_boundary, tail_boundary, total_balance, moment_balance]):
        model = TrimBlock(
            ww=tensor(a[:2, :2], over=(WT, WS)),
            wt=tensor(a[:2, 2:], over=(WT, TS)),
            tw=tensor(a[2:, :2], over=(TT, WS)),
            tt=tensor(a[2:, 2:], over=(TT, TS)),
            wing_width=tensor(widths[:2], over=WS),
            tail_width=tensor(widths[2:], over=TS),
            wing_arm=tensor(arms[:2], over=WS),
            tail_arm=tensor(arms[2:], over=TS),
            wing_rhs=tensor(right[:2], over=WT),
            tail_rhs=tensor(right[2:4], over=TT),
            total=float(right[4]),
            moment=float(right[5]),
        )
        result = getattr(model, output)

    reference = np.linalg.solve(coefficient, right)
    if output == "wing":
        assert isinstance(result, Tensor)
        assert result.domain == domain(WS)
        np.testing.assert_allclose(result.values, reference[:2], atol=1e-6)
    elif output == "tail":
        assert isinstance(result, Tensor)
        assert result.domain == domain(TS)
        np.testing.assert_allclose(result.values, reference[2:4], atol=1e-6)
    elif output == "alpha":
        assert float(result) == pytest.approx(reference[4], abs=1e-6)
    else:
        assert float(result) == pytest.approx(reference[5], abs=1e-6)
