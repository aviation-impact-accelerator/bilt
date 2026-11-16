"""Four-panel prescribed wing-tail aerodynamic coefficients and flight inputs."""

from typing import NotRequired, TypedDict

from bilt import Quantity, QuantityTensor, tensor
from bilt.units import degree, kilogram, meter, radian, second
from bilt.units.units import newton

from .model import TAIL_SOURCE, TAIL_TARGET, WING_SOURCE, WING_TARGET, WingTailTrim


class _AircraftInputs(TypedDict):
    """Known physical inputs; missing state variables remain acausal unknowns."""

    wing_from_wing: QuantityTensor
    wing_from_tail: QuantityTensor
    tail_from_wing: QuantityTensor
    tail_from_tail: QuantityTensor
    wing_width: QuantityTensor
    tail_width: QuantityTensor
    wing_arm: QuantityTensor
    tail_arm: QuantityTensor
    wing_twist: QuantityTensor
    tail_twist: QuantityTensor
    tail_control: QuantityTensor
    density: Quantity
    speed: Quantity
    gravity: Quantity
    mass: NotRequired[Quantity]
    alpha: NotRequired[Quantity]
    elevator: NotRequired[Quantity]
    pitch_moment: NotRequired[Quantity]


def example_aircraft(
    *,
    mass: Quantity | None = None,
    alpha: Quantity | None = None,
    elevator: Quantity | None = None,
    pitch_moment: Quantity | None = None,
) -> WingTailTrim:
    """Four independent source panels with prescribed cross-surface influence.

    Not a geometry-derived lifting-line method. Unsymmetric coefficients
    make all four panel circulations distinct and independently testable.
    """
    known: _AircraftInputs = {
        "wing_from_wing": tensor(
            [[-0.35, -0.07], [-0.065, -0.34]],
            over=(WING_TARGET, WING_SOURCE),
            unit=meter**-1,
        ),
        "wing_from_tail": tensor(
            [[-0.015, -0.005], [-0.008, -0.014]],
            over=(WING_TARGET, TAIL_SOURCE),
            unit=meter**-1,
        ),
        "tail_from_wing": tensor(
            [[-0.06, -0.045], [-0.05, -0.065]],
            over=(TAIL_TARGET, WING_SOURCE),
            unit=meter**-1,
        ),
        "tail_from_tail": tensor(
            [[-0.59, -0.075], [-0.065, -0.61]],
            over=(TAIL_TARGET, TAIL_SOURCE),
            unit=meter**-1,
        ),
        "wing_width": tensor([4.1, 3.9], over=WING_SOURCE, unit=meter),
        "tail_width": tensor([2.1, 1.9], over=TAIL_SOURCE, unit=meter),
        "wing_arm": tensor([0.5, 0.55], over=WING_SOURCE, unit=meter),
        "tail_arm": tensor([-3.9, -4.1], over=TAIL_SOURCE, unit=meter),
        "wing_twist": tensor([0.005, -0.003], over=WING_TARGET, unit=radian),
        "tail_twist": tensor([0.004, -0.002], over=TAIL_TARGET, unit=radian),
        "tail_control": tensor([1.0, 1.0], over=TAIL_TARGET, unit=radian / radian),
        "density": 1.225 * kilogram / meter**3,
        "speed": 70.0 * meter / second,
        "gravity": 9.81 * meter / second**2,
    }
    if mass is not None:
        known["mass"] = mass
    if alpha is not None:
        known["alpha"] = alpha
    if elevator is not None:
        known["elevator"] = elevator
    if pitch_moment is not None:
        known["pitch_moment"] = pitch_moment
    return WingTailTrim(**known)


def trim_case(mass_kg: float = 550.0) -> WingTailTrim:
    """Fix weight and zero pitching moment, leaving both trim angles unknown."""
    return example_aircraft(mass=mass_kg * kilogram, pitch_moment=0.0 * newton * meter)


def forward_case(alpha_deg: float = 3.0, elevator_deg: float = 0.0) -> WingTailTrim:
    """Fix both angles and compute independently distributed panel circulations."""
    return example_aircraft(alpha=alpha_deg * degree, elevator=elevator_deg * degree)
