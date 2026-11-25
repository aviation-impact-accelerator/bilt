"""Dimensional lifting-surface geometry and horseshoe-vortex equations."""

import math
from dataclasses import dataclass
from typing import Annotated, TypedDict

import numpy as np

from bilt import Implementation, Interface, Quantity, QuantityTensor, dim, equation, field, system, tensor
from bilt.units import fraction, meter, radian
from bilt.units.dimensions import DIMENSIONLESS, LENGTH

from .model import TAIL_SOURCE, TAIL_TARGET, WING_SOURCE, WING_TARGET

_SINGULARITY_TOLERANCE_M = 1e-12
_SINGULARITY_CROSS_SQUARED = _SINGULARITY_TOLERANCE_M**4

PANEL = dim("panel")
CARTESIAN = dim("cartesian")
SEGMENT = dim("segment")
VORTEX_SOURCE = dim("vortex_source")
VORTEX_TARGET = dim("vortex_target")

_LENGTH_SCALE = 1.0 * meter

type Length = Annotated[Quantity, LENGTH]
type Area = Annotated[Quantity, LENGTH**2]
type Ratio = Annotated[Quantity, DIMENSIONLESS]
type PanelLength = Annotated[QuantityTensor, LENGTH]
type PanelRatio = Annotated[QuantityTensor, DIMENSIONLESS]
type SquaredLength = Annotated[QuantityTensor, LENGTH**2]
type FourthPowerLength = Annotated[QuantityTensor, LENGTH**4]
type Influence = Annotated[QuantityTensor, LENGTH**-1]


@system
class RectangularSurface(metaclass=Interface):
    """Flat rectangular lifting surface; x aft, y right, z up."""

    span: Length = field()
    chord: Length = field()
    leading_edge_x: Length = field()
    height: Length = field()
    root_twist: Ratio = field()
    tip_twist: Ratio = field()
    area: Area = field()
    aspect_ratio: Ratio = field()
    bound_x: Length = field()
    target_x: Length = field()


@equation
def surface_area(area: Area, span: Length, chord: Length) -> bool:
    """Rectangular planform area."""
    return area == span * chord


@equation
def surface_aspect_ratio(aspect_ratio: Ratio, span: Length, area: Area) -> bool:
    """Span squared over planform area."""
    return aspect_ratio * area == span**2


@equation
def bound_position(bound_x: Length, leading_edge_x: Length, chord: Length) -> bool:
    """Place the bound vortex at quarter chord."""
    return bound_x == leading_edge_x + 0.25 * chord


@equation
def target_position(target_x: Length, leading_edge_x: Length, chord: Length) -> bool:
    """Place the collocation point at three-quarter chord."""
    return target_x == leading_edge_x + 0.75 * chord


SURFACE_MODEL = Implementation(
    RectangularSurface, [surface_area, surface_aspect_ratio, bound_position, target_position]
)


@system
class AircraftGeometry(metaclass=Interface):
    """Wing and horizontal-tail geometry with a fixed downstream wake."""

    wing_span: Length = field()
    wing_chord: Length = field()
    wing_leading_edge_x: Length = field()
    wing_height: Length = field()
    wing_root_twist: Ratio = field()
    wing_tip_twist: Ratio = field()
    wing_area: Area = field()
    tail_span: Length = field()
    tail_chord: Length = field()
    tail_leading_edge_x: Length = field()
    tail_height: Length = field()
    tail_root_twist: Ratio = field()
    tail_tip_twist: Ratio = field()
    tail_area: Area = field()
    cg_x: Length = field()
    wake_length: Length = field()


@equation
def wing_span_connection(wing_span: Length, wing: RectangularSurface) -> bool:
    """Connect the wing span."""
    return wing_span == wing.span


@equation
def wing_chord_connection(wing_chord: Length, wing: RectangularSurface) -> bool:
    """Connect the wing chord."""
    return wing_chord == wing.chord


@equation
def wing_area_connection(wing_area: Area, wing: RectangularSurface) -> bool:
    """Expose the wing area."""
    return wing_area == wing.area


@equation
def tail_span_connection(tail_span: Length, tail: RectangularSurface) -> bool:
    """Connect the tail span."""
    return tail_span == tail.span


@equation
def tail_chord_connection(tail_chord: Length, tail: RectangularSurface) -> bool:
    """Connect the tail chord."""
    return tail_chord == tail.chord


@equation
def tail_area_connection(tail_area: Area, tail: RectangularSurface) -> bool:
    """Expose the tail area."""
    return tail_area == tail.area


GEOMETRY_MODEL = Implementation(
    AircraftGeometry,
    [
        wing_span_connection,
        wing_chord_connection,
        wing_area_connection,
        tail_span_connection,
        tail_chord_connection,
        tail_area_connection,
    ],
    internal_implementations={"wing": SURFACE_MODEL, "tail": SURFACE_MODEL},
)


@dataclass(frozen=True, slots=True)
class PanelResolution:
    """Numerical discretisation settings, not physical geometry."""

    wing: int = 2
    tail: int = 2

    def __post_init__(self) -> None:
        """Require a positive number of panels on both surfaces."""
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 1 for count in (self.wing, self.tail)):
            msg = "Panel count must be a positive integer."
            raise ValueError(msg)


DEFAULT_RESOLUTION = PanelResolution()


def example_geometry(*, wake_length: Quantity = 400.0 * meter) -> AircraftGeometry:
    """Return the physical aircraft independently of its discretisation."""
    return AircraftGeometry(
        wing_span=12.0 * meter,
        wing_chord=1.5 * meter,
        wing_leading_edge_x=0.0 * meter,
        wing_height=0.0 * meter,
        wing_root_twist=0.002 * radian,
        wing_tip_twist=-0.008 * radian,
        tail_span=4.5 * meter,
        tail_chord=0.9 * meter,
        tail_leading_edge_x=5.5 * meter,
        tail_height=0.5 * meter,
        tail_root_twist=0.0 * radian,
        tail_tip_twist=-0.003 * radian,
        cg_x=1.0 * meter,
        wake_length=wake_length,
    )


@system
class SurfacePanels(metaclass=Interface):
    """Dimensional spanwise panel geometry over a named panel axis."""

    span: Length = field()
    bound_x: Length = field()
    target_x: Length = field()
    height: Length = field()
    cg_x: Length = field()
    root_twist: Ratio = field()
    tip_twist: Ratio = field()
    left_fraction: PanelRatio = field()
    right_fraction: PanelRatio = field()
    twist_fraction: PanelRatio = field()
    left_y: PanelLength = field()
    right_y: PanelLength = field()
    centre_y: PanelLength = field()
    width: PanelLength = field()
    arm: Length = field()
    twist: PanelRatio = field()


@equation
def left_position(left_y: PanelLength, span: Length, left_fraction: PanelRatio) -> bool:
    """Scale the normalized left panel edge by span."""
    return left_y == span * left_fraction


@equation
def right_position(right_y: PanelLength, span: Length, right_fraction: PanelRatio) -> bool:
    """Scale the normalized right panel edge by span."""
    return right_y == span * right_fraction


@equation
def centre_position(centre_y: PanelLength, left_y: PanelLength, right_y: PanelLength) -> bool:
    """Collocation points lie at the spanwise panel centres."""
    return centre_y == 0.5 * (left_y + right_y)


@equation
def panel_width(width: PanelLength, left_y: PanelLength, right_y: PanelLength) -> bool:
    """Width is the distance between consecutive panel edges."""
    return width == right_y - left_y


@equation
def moment_arm(arm: Length, cg_x: Length, bound_x: Length) -> bool:
    """Positive lift ahead of the CG gives positive pitch moment."""
    return arm == cg_x - bound_x


@equation
def spanwise_twist_fraction(
    twist_fraction: PanelRatio, left_fraction: PanelRatio, right_fraction: PanelRatio
) -> bool:
    """Absolute span fraction at the panel centre."""
    return twist_fraction == ((left_fraction + right_fraction) ** 2) ** 0.5


@equation
def panel_twist(twist: PanelRatio, root_twist: Ratio, tip_twist: Ratio, twist_fraction: PanelRatio) -> bool:
    """Linearly interpolate twist using absolute normalized span."""
    return twist == root_twist + (tip_twist - root_twist) * twist_fraction


PANEL_MODEL = Implementation(
    SurfacePanels,
    [left_position, right_position, centre_position, panel_width, moment_arm, spanwise_twist_fraction, panel_twist],
)


def _validate_surface(surface: RectangularSurface, panels: int) -> None:
    values = tuple(
        value.convert_to(meter).value
        for value in (surface.span, surface.chord, surface.leading_edge_x, surface.height)
    )
    if not np.isfinite(values).all():
        msg = "Lifting-surface geometry must contain finite coordinates."
        raise ValueError(msg)
    invalid_count = isinstance(panels, bool) or not isinstance(panels, int) or panels < 1
    if values[0] <= 0 or values[1] <= 0 or invalid_count:
        msg = "Lifting-surface span, chord and panel count must be positive."
        raise ValueError(msg)
    if not np.isfinite((surface.root_twist.convert_to(radian).value, surface.tip_twist.convert_to(radian).value)).all():
        msg = "Lifting-surface twist must be finite."
        raise ValueError(msg)


def discretise(surface: RectangularSurface, *, cg_x: Quantity, panels: int) -> SurfacePanels:
    """Generate the numerical panel axis and evaluate dimensional geometry."""
    with SURFACE_MODEL:
        _validate_surface(surface, panels)
        edges = np.linspace(-0.5, 0.5, panels + 1)
        return SurfacePanels(
            span=surface.span,
            bound_x=surface.bound_x,
            target_x=surface.target_x,
            height=surface.height,
            cg_x=cg_x,
            root_twist=surface.root_twist,
            tip_twist=surface.tip_twist,
            left_fraction=tensor(edges[:-1], over=PANEL, unit=fraction),
            right_fraction=tensor(edges[1:], over=PANEL, unit=fraction),
        )


def _points(x: Quantity, y: QuantityTensor, z: Quantity) -> np.ndarray:
    values = y.convert_to(meter).values
    return np.column_stack(
        (np.full(values.size, x.convert_to(meter).value), values, np.full(values.size, z.convert_to(meter).value))
    )


@system
class VortexInfluence(metaclass=Interface):
    """Finite vortex segments evaluated at collocation points.

    Position tensors carry the Cartesian axis and may have independent source,
    target and segment axes. The result is vertical velocity per circulation.
    """

    point: PanelLength = field()
    start: PanelLength = field()
    end: PanelLength = field()
    r1: PanelLength = field()
    r2: PanelLength = field()
    direction: PanelLength = field()
    cross_x: SquaredLength = field()
    cross_y: SquaredLength = field()
    cross_z: SquaredLength = field()
    cross_squared: FourthPowerLength = field()
    norm1: PanelLength = field()
    norm2: PanelLength = field()
    endpoint_factor: PanelLength = field()
    segment_influence: Influence = field()
    influence: Influence = field()


@equation
def first_displacement(r1: PanelLength, point: PanelLength, start: PanelLength) -> bool:
    """Vector from the first endpoint to the collocation point."""
    return r1 == point - start


@equation
def second_displacement(r2: PanelLength, point: PanelLength, end: PanelLength) -> bool:
    """Vector from the second endpoint to the collocation point."""
    return r2 == point - end


@equation
def segment_direction(direction: PanelLength, start: PanelLength, end: PanelLength) -> bool:
    """Vector along the finite vortex."""
    return direction == end - start


@equation
def cross_product_x(cross_x: SquaredLength, r1: PanelLength, r2: PanelLength) -> bool:
    """First Cartesian component of the displacement cross product."""
    return cross_x == r1.isel(CARTESIAN, 1) * r2.isel(CARTESIAN, 2) - r1.isel(CARTESIAN, 2) * r2.isel(CARTESIAN, 1)


@equation
def cross_product_y(cross_y: SquaredLength, r1: PanelLength, r2: PanelLength) -> bool:
    """Second Cartesian component of the displacement cross product."""
    return cross_y == r1.isel(CARTESIAN, 2) * r2.isel(CARTESIAN, 0) - r1.isel(CARTESIAN, 0) * r2.isel(CARTESIAN, 2)


@equation
def cross_product_z(cross_z: SquaredLength, r1: PanelLength, r2: PanelLength) -> bool:
    """Vertical component of the displacement cross product."""
    return cross_z == r1.isel(CARTESIAN, 0) * r2.isel(CARTESIAN, 1) - r1.isel(CARTESIAN, 1) * r2.isel(CARTESIAN, 0)


@equation
def cross_magnitude_squared(
    cross_squared: FourthPowerLength, cross_x: SquaredLength, cross_y: SquaredLength, cross_z: SquaredLength
) -> bool:
    """Squared norm of the cross product."""
    return cross_squared == cross_x**2 + cross_y**2 + cross_z**2


@equation
def first_distance(norm1: PanelLength, r1: PanelLength) -> bool:
    """Euclidean distance, using a dimensionless argument for the square root."""
    return norm1 == _LENGTH_SCALE * ((r1 / _LENGTH_SCALE) ** 2).sum(CARTESIAN) ** 0.5


@equation
def second_distance(norm2: PanelLength, r2: PanelLength) -> bool:
    """Euclidean distance to the second endpoint."""
    return norm2 == _LENGTH_SCALE * ((r2 / _LENGTH_SCALE) ** 2).sum(CARTESIAN) ** 0.5


@equation
def endpoint_projection(  # noqa: PLR0913, PLR0917 - physical equation ports.
    endpoint_factor: PanelLength,
    direction: PanelLength,
    r1: PanelLength,
    r2: PanelLength,
    norm1: PanelLength,
    norm2: PanelLength,
) -> bool:
    """Dot the segment direction with the difference of endpoint unit vectors."""
    return endpoint_factor == (direction * (r1 / norm1 - r2 / norm2)).sum(CARTESIAN)


@equation
def biot_savart(
    segment_influence: Influence,
    cross_z: SquaredLength,
    endpoint_factor: PanelLength,
    cross_squared: FourthPowerLength,
) -> bool:
    """Vertical finite-segment velocity per unit circulation."""
    return segment_influence == cross_z * endpoint_factor / (4.0 * math.pi * cross_squared)


@equation
def integrate_segments(influence: Influence, segment_influence: Influence) -> bool:
    """Superpose the finite segments of the horseshoe."""
    return influence == segment_influence.sum(SEGMENT)


VORTEX_MODEL = Implementation(
    VortexInfluence,
    [
        first_displacement,
        second_displacement,
        segment_direction,
        cross_product_x,
        cross_product_y,
        cross_product_z,
        cross_magnitude_squared,
        first_distance,
        second_distance,
        endpoint_projection,
        biot_savart,
        integrate_segments,
    ],
)


def evaluate_vortices(point: QuantityTensor, start: QuantityTensor, end: QuantityTensor) -> QuantityTensor:
    """Evaluate the model, rejecting singular geometries before division."""
    with VORTEX_MODEL:
        vortex = VortexInfluence(point=point, start=start, end=end)
        cross_squared = vortex.cross_squared.convert_to(meter**4).values
        norm1 = vortex.norm1.convert_to(meter).values
        norm2 = vortex.norm2.convert_to(meter).values
        if (
            np.any(cross_squared <= _SINGULARITY_CROSS_SQUARED)
            or np.any(norm1 <= _SINGULARITY_TOLERANCE_M)
            or np.any(norm2 <= _SINGULARITY_TOLERANCE_M)
        ):
            msg = "Vortex evaluation point lies on or too near a filament."
            raise ValueError(msg)
        return vortex.influence


def influence_block(targets: SurfacePanels, sources: SurfacePanels, *, wake_length: Quantity) -> np.ndarray:
    """Evaluate every target/source interaction using named Bilt tensor axes."""
    with PANEL_MODEL:
        left = _points(sources.bound_x, sources.left_y, sources.height)
        right = _points(sources.bound_x, sources.right_y, sources.height)
        collocation = _points(targets.target_x, targets.centre_y, targets.height)

    # The three oriented segments are numerical topology, not aerodynamic laws.
    wake = np.array([wake_length.convert_to(meter).value, 0.0, 0.0])
    starts = np.stack((left + wake, left, right), axis=1)
    ends = np.stack((left, right, right + wake), axis=1)
    result = evaluate_vortices(
        tensor(collocation, over=(VORTEX_TARGET, CARTESIAN), unit=meter),
        tensor(starts, over=(VORTEX_SOURCE, SEGMENT, CARTESIAN), unit=meter),
        tensor(ends, over=(VORTEX_SOURCE, SEGMENT, CARTESIAN), unit=meter),
    )
    order = tuple(result.domain.dims.index(axis) for axis in (VORTEX_TARGET, VORTEX_SOURCE))
    return result.convert_to(meter**-1).values.transpose(order)


class InfluenceInputs(TypedDict):
    """Named dimensional inputs at the numerical kernel / trim model boundary."""

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


def calculate_influences(
    geometry: AircraftGeometry, *, resolution: PanelResolution = DEFAULT_RESOLUTION
) -> InfluenceInputs:
    """Evaluate the geometry and supply named influence and load tensors."""
    with GEOMETRY_MODEL:
        wing_surface = RectangularSurface(
            span=geometry.wing_span,
            chord=geometry.wing_chord,
            leading_edge_x=geometry.wing_leading_edge_x,
            height=geometry.wing_height,
            root_twist=geometry.wing_root_twist,
            tip_twist=geometry.wing_tip_twist,
        )
        tail_surface = RectangularSurface(
            span=geometry.tail_span,
            chord=geometry.tail_chord,
            leading_edge_x=geometry.tail_leading_edge_x,
            height=geometry.tail_height,
            root_twist=geometry.tail_root_twist,
            tip_twist=geometry.tail_tip_twist,
        )
        cg_x, wake = geometry.cg_x, geometry.wake_length
    cg_m = cg_x.convert_to(meter).value
    wake_m = wake.convert_to(meter).value
    if not np.isfinite((cg_m, wake_m)).all() or wake_m <= 0:
        msg = "CG coordinate must be finite and wake length positive."
        raise ValueError(msg)
    if tail_surface.leading_edge_x.convert_to(meter).value <= wing_surface.leading_edge_x.convert_to(meter).value:
        msg = "The horizontal tail must be downstream of the wing."
        raise ValueError(msg)
    wing = discretise(wing_surface, cg_x=cg_x, panels=resolution.wing)
    tail = discretise(tail_surface, cg_x=cg_x, panels=resolution.tail)
    with PANEL_MODEL:
        return {
            "wing_from_wing": tensor(
                influence_block(wing, wing, wake_length=wake), over=(WING_TARGET, WING_SOURCE), unit=meter**-1
            ),
            "wing_from_tail": tensor(
                influence_block(wing, tail, wake_length=wake), over=(WING_TARGET, TAIL_SOURCE), unit=meter**-1
            ),
            "tail_from_wing": tensor(
                influence_block(tail, wing, wake_length=wake), over=(TAIL_TARGET, WING_SOURCE), unit=meter**-1
            ),
            "tail_from_tail": tensor(
                influence_block(tail, tail, wake_length=wake), over=(TAIL_TARGET, TAIL_SOURCE), unit=meter**-1
            ),
            "wing_width": tensor(wing.width.convert_to(meter).values, over=WING_SOURCE, unit=meter),
            "tail_width": tensor(tail.width.convert_to(meter).values, over=TAIL_SOURCE, unit=meter),
            "wing_arm": tensor(
                np.full(resolution.wing, wing.arm.convert_to(meter).value), over=WING_SOURCE, unit=meter
            ),
            "tail_arm": tensor(
                np.full(resolution.tail, tail.arm.convert_to(meter).value), over=TAIL_SOURCE, unit=meter
            ),
            "wing_twist": tensor(wing.twist.convert_to(radian).values, over=WING_TARGET, unit=radian),
            "tail_twist": tensor(tail.twist.convert_to(radian).values, over=TAIL_TARGET, unit=radian),
        }
