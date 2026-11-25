"""Geometry-derived wing-and-tail influence tensors and flight conditions."""

from typing import NotRequired

from bilt import Quantity, QuantityTensor, tensor
from bilt.units import degree, kilogram, meter, radian, second
from bilt.units.units import newton

from .geometry import AircraftGeometry, InfluenceInputs, PanelResolution, calculate_influences, example_geometry
from .model import TAIL_TARGET, WingTailTrim


class _AircraftInputs(InfluenceInputs):
    """Known physical inputs; omitted fields are acausal unknowns."""

    tail_control: QuantityTensor
    density: Quantity
    speed: Quantity
    gravity: Quantity
    mass: NotRequired[Quantity]
    lift: NotRequired[Quantity]
    alpha: NotRequired[Quantity]
    elevator: NotRequired[Quantity]
    pitch_moment: NotRequired[Quantity]


def example_aircraft(  # noqa: PLR0913 - independent boundary inputs and discretisation.
    *,
    geometry: AircraftGeometry | None = None,
    resolution: PanelResolution = PanelResolution(),
    mass: Quantity | None = None,
    lift: Quantity | None = None,
    alpha: Quantity | None = None,
    elevator: Quantity | None = None,
    pitch_moment: Quantity | None = None,
) -> WingTailTrim:
    """Construct a geometry-derived linear wing-and-tail aerodynamic model.

    Horseshoe vortex influences are geometry input data. The aerodynamic and
    trim equations themselves remain expressed entirely using Bilt relations.
    """
    geometry = geometry if geometry is not None else example_geometry()
    coefficients = calculate_influences(geometry, resolution=resolution)
    known: _AircraftInputs = {
        **coefficients,
        "tail_control": tensor([1.0] * resolution.tail, over=TAIL_TARGET, unit=radian / radian),
        "density": 1.225 * kilogram / meter**3,
        "speed": 70.0 * meter / second,
        "gravity": 9.81 * meter / second**2,
    }
    if mass is not None:
        known["mass"] = mass
    if lift is not None:
        known["lift"] = lift
    if alpha is not None:
        known["alpha"] = alpha
    if elevator is not None:
        known["elevator"] = elevator
    if pitch_moment is not None:
        known["pitch_moment"] = pitch_moment
    return WingTailTrim(**known)


def trim_case(
    mass_kg: float = 550.0,
    *,
    geometry: AircraftGeometry | None = None,
    resolution: PanelResolution = PanelResolution(),
) -> WingTailTrim:
    """Fix weight and zero pitching moment, solving both trim angles."""
    return example_aircraft(
        geometry=geometry,
        resolution=resolution,
        mass=mass_kg * kilogram,
        pitch_moment=0.0 * newton * meter,
    )


def forward_case(
    alpha_deg: float = 3.0,
    elevator_deg: float = 0.0,
    *,
    geometry: AircraftGeometry | None = None,
    resolution: PanelResolution = PanelResolution(),
) -> WingTailTrim:
    """Fix the two angles and solve for the full circulation distribution."""
    return example_aircraft(
        geometry=geometry,
        resolution=resolution,
        alpha=alpha_deg * degree,
        elevator=elevator_deg * degree,
    )
