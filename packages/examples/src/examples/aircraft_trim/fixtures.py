"""Geometry-derived wing-and-tail influence tensors and flight conditions."""

from typing import NotRequired, TypedDict

from bilt import Quantity, QuantityTensor, tensor
from bilt.units import degree, kilogram, meter, radian, second
from bilt.units.units import newton

from .geometry import AircraftGeometry, calculate_influences, example_geometry
from .model import TAIL_SOURCE, TAIL_TARGET, WING_SOURCE, WING_TARGET, WingTailTrim


class _AircraftInputs(TypedDict):
    """Known physical inputs; omitted fields are acausal unknowns."""

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
    geometry: AircraftGeometry | None = None,
    mass: Quantity | None = None,
    alpha: Quantity | None = None,
    elevator: Quantity | None = None,
    pitch_moment: Quantity | None = None,
) -> WingTailTrim:
    """Construct a geometry-derived linear wing-and-tail aerodynamic model.

    Horseshoe vortex influences are geometry input data. The aerodynamic and
    trim equations themselves remain expressed entirely using Bilt relations.
    """
    geometry = geometry if geometry is not None else example_geometry()
    coefficients = calculate_influences(geometry)
    known: _AircraftInputs = {
        "wing_from_wing": tensor(
            coefficients.wing_from_wing, over=(WING_TARGET, WING_SOURCE), unit=meter**-1
        ),
        "wing_from_tail": tensor(
            coefficients.wing_from_tail, over=(WING_TARGET, TAIL_SOURCE), unit=meter**-1
        ),
        "tail_from_wing": tensor(
            coefficients.tail_from_wing, over=(TAIL_TARGET, WING_SOURCE), unit=meter**-1
        ),
        "tail_from_tail": tensor(
            coefficients.tail_from_tail, over=(TAIL_TARGET, TAIL_SOURCE), unit=meter**-1
        ),
        "wing_width": tensor(coefficients.wing.width_m, over=WING_SOURCE, unit=meter),
        "tail_width": tensor(coefficients.tail.width_m, over=TAIL_SOURCE, unit=meter),
        "wing_arm": tensor(coefficients.wing.arm_m, over=WING_SOURCE, unit=meter),
        "tail_arm": tensor(coefficients.tail.arm_m, over=TAIL_SOURCE, unit=meter),
        "wing_twist": tensor(coefficients.wing.twist_rad, over=WING_TARGET, unit=radian),
        "tail_twist": tensor(coefficients.tail.twist_rad, over=TAIL_TARGET, unit=radian),
        "tail_control": tensor(
            [1.0] * geometry.tail.panels, over=TAIL_TARGET, unit=radian / radian
        ),
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


def trim_case(mass_kg: float = 550.0, *, geometry: AircraftGeometry | None = None) -> WingTailTrim:
    """Fix weight and zero pitching moment, solving both trim angles."""
    return example_aircraft(
        geometry=geometry,
        mass=mass_kg * kilogram,
        pitch_moment=0.0 * newton * meter,
    )


def forward_case(
    alpha_deg: float = 3.0,
    elevator_deg: float = 0.0,
    *,
    geometry: AircraftGeometry | None = None,
) -> WingTailTrim:
    """Fix the two angles and solve for the full circulation distribution."""
    return example_aircraft(
        geometry=geometry,
        alpha=alpha_deg * degree,
        elevator=elevator_deg * degree,
    )
