"""Input construction for the composed aircraft example."""

from typing import TYPE_CHECKING, NotRequired, TypedDict

import numpy as np

from bilt import tensor
from bilt.units import degree, fraction, kilogram, meter, newton, second

from .aircraft import Aircraft
from .geometry import (
    CARTESIAN,
    SEGMENT,
    AircraftGeometry,
    PanelResolution,
    default_resolution,
    example_geometry,
    validate_panel_count,
)
from .model import TAIL_SOURCE, TAIL_TARGET, WING_SOURCE, WING_TARGET

if TYPE_CHECKING:
    from bilt import Quantity, QuantityTensor


class _AircraftInputs(TypedDict):
    """Known physical inputs and discrete axes; omit unknown flight conditions."""

    wing_span: Quantity
    wing_chord: Quantity
    wing_leading_edge_x: Quantity
    wing_height: Quantity
    wing_root_twist: Quantity
    wing_tip_twist: Quantity
    tail_span: Quantity
    tail_chord: Quantity
    tail_leading_edge_x: Quantity
    tail_height: Quantity
    tail_root_twist: Quantity
    tail_tip_twist: Quantity
    cg_x: Quantity
    wake_length: Quantity
    wing_panel_count: Quantity
    wing_source_index: QuantityTensor
    wing_target_index: QuantityTensor
    tail_panel_count: Quantity
    tail_source_index: QuantityTensor
    tail_target_index: QuantityTensor
    x_mask: QuantityTensor
    y_mask: QuantityTensor
    z_mask: QuantityTensor
    left_segment: QuantityTensor
    bound_segment: QuantityTensor
    right_segment: QuantityTensor
    tail_control: Quantity
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
    resolution: PanelResolution | None = None,
    mass: Quantity | None = None,
    lift: Quantity | None = None,
    alpha: Quantity | None = None,
    elevator: Quantity | None = None,
    pitch_moment: Quantity | None = None,
) -> Aircraft:
    """Supply physical inputs and named axes, without evaluating any aerodynamics."""
    geometry = geometry if geometry is not None else example_geometry()
    resolution = resolution if resolution is not None else default_resolution()
    validate_panel_count(resolution.wing)
    validate_panel_count(resolution.tail)
    known: _AircraftInputs = {
        "wing_span": geometry.wing_span,
        "wing_chord": geometry.wing_chord,
        "wing_leading_edge_x": geometry.wing_leading_edge_x,
        "wing_height": geometry.wing_height,
        "wing_root_twist": geometry.wing_root_twist,
        "wing_tip_twist": geometry.wing_tip_twist,
        "tail_span": geometry.tail_span,
        "tail_chord": geometry.tail_chord,
        "tail_leading_edge_x": geometry.tail_leading_edge_x,
        "tail_height": geometry.tail_height,
        "tail_root_twist": geometry.tail_root_twist,
        "tail_tip_twist": geometry.tail_tip_twist,
        "cg_x": geometry.cg_x,
        "wake_length": geometry.wake_length,
        "wing_panel_count": float(resolution.wing) * fraction,
        "tail_panel_count": float(resolution.tail) * fraction,
        "wing_source_index": tensor(np.arange(resolution.wing), over=WING_SOURCE, unit=fraction),
        "wing_target_index": tensor(np.arange(resolution.wing), over=WING_TARGET, unit=fraction),
        "tail_source_index": tensor(np.arange(resolution.tail), over=TAIL_SOURCE, unit=fraction),
        "tail_target_index": tensor(np.arange(resolution.tail), over=TAIL_TARGET, unit=fraction),
        "x_mask": tensor([1, 0, 0], over=CARTESIAN, unit=fraction),
        "y_mask": tensor([0, 1, 0], over=CARTESIAN, unit=fraction),
        "z_mask": tensor([0, 0, 1], over=CARTESIAN, unit=fraction),
        "left_segment": tensor([1, 0, 0], over=SEGMENT, unit=fraction),
        "bound_segment": tensor([0, 1, 0], over=SEGMENT, unit=fraction),
        "right_segment": tensor([0, 0, 1], over=SEGMENT, unit=fraction),
        "tail_control": 1.0 * fraction,
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
    return Aircraft(**known)


def trim_case(
    mass_kg: float = 550.0,
    *,
    geometry: AircraftGeometry | None = None,
    resolution: PanelResolution | None = None,
) -> Aircraft:
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
    resolution: PanelResolution | None = None,
) -> Aircraft:
    """Fix the two angles and solve for the full circulation distribution."""
    return example_aircraft(
        geometry=geometry,
        resolution=resolution,
        alpha=alpha_deg * degree,
        elevator=elevator_deg * degree,
    )
