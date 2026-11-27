"""Dimensional lifting-surface geometry and horseshoe-vortex equations."""

import math
from typing import TYPE_CHECKING, Annotated, TypedDict

import numpy as np

from bilt import Implementation, Interface, Quantity, QuantityTensor, dim, equation, field, system, tensor, transform
from bilt.units import fraction, meter, radian
from bilt.units.dimensions import DIMENSIONLESS, LENGTH

from .model import TAIL_SOURCE, TAIL_TARGET, WING_SOURCE, WING_TARGET

if TYPE_CHECKING:
    from bilt import Dim

PANEL = dim("panel")
CARTESIAN = dim("cartesian")
SEGMENT = dim("segment")

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


@equation
def wing_leading_edge_x_connection(wing_leading_edge_x: Length, wing: RectangularSurface) -> bool:
    """Connect the wing leading edge x."""
    return wing_leading_edge_x == wing.leading_edge_x


@equation
def wing_height_connection(wing_height: Length, wing: RectangularSurface) -> bool:
    """Connect the wing height."""
    return wing_height == wing.height


@equation
def wing_root_twist_connection(wing_root_twist: Ratio, wing: RectangularSurface) -> bool:
    """Connect the wing root twist."""
    return wing_root_twist == wing.root_twist


@equation
def wing_tip_twist_connection(wing_tip_twist: Ratio, wing: RectangularSurface) -> bool:
    """Connect the wing tip twist."""
    return wing_tip_twist == wing.tip_twist


@equation
def tail_leading_edge_x_connection(tail_leading_edge_x: Length, tail: RectangularSurface) -> bool:
    """Connect the tail leading edge x."""
    return tail_leading_edge_x == tail.leading_edge_x


@equation
def tail_height_connection(tail_height: Length, tail: RectangularSurface) -> bool:
    """Connect the tail height."""
    return tail_height == tail.height


@equation
def tail_root_twist_connection(tail_root_twist: Ratio, tail: RectangularSurface) -> bool:
    """Connect the tail root twist."""
    return tail_root_twist == tail.root_twist


@equation
def tail_tip_twist_connection(tail_tip_twist: Ratio, tail: RectangularSurface) -> bool:
    """Connect the tail tip twist."""
    return tail_tip_twist == tail.tip_twist


GEOMETRY_MODEL = Implementation(
    AircraftGeometry,
    [
        wing_span_connection,
        wing_chord_connection,
        wing_area_connection,
        tail_span_connection,
        tail_chord_connection,
        tail_area_connection,
        wing_leading_edge_x_connection,
        wing_height_connection,
        wing_root_twist_connection,
        wing_tip_twist_connection,
        tail_leading_edge_x_connection,
        tail_height_connection,
        tail_root_twist_connection,
        tail_tip_twist_connection,
    ],
    internal_implementations={"wing": SURFACE_MODEL, "tail": SURFACE_MODEL},
)


@system
class PanelResolution(metaclass=Interface):
    """Discrete panel counts for the two lifting surfaces."""

    wing: int = field()
    tail: int = field()


def validate_panel_count(count: int) -> None:
    """Require an integer before constructing the named panel axis."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        msg = "Panel count must be a positive integer."
        raise ValueError(msg)


def default_resolution() -> PanelResolution:
    """Return the default discretisation without sharing a system instance."""
    return PanelResolution(wing=2, tail=2)


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
    panel_index: PanelRatio = field()
    panel_count: Ratio = field()
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
def left_edge_fraction(left_fraction: PanelRatio, panel_index: PanelRatio, panel_count: Ratio) -> bool:
    """Normalised left edge of each uniformly spaced panel."""
    return left_fraction == panel_index / panel_count - 0.5


@equation
def right_edge_fraction(right_fraction: PanelRatio, panel_index: PanelRatio, panel_count: Ratio) -> bool:
    """Normalised right edge of each uniformly spaced panel."""
    return right_fraction == (panel_index + 1.0) / panel_count - 0.5


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
def spanwise_twist_fraction(twist_fraction: Ratio, left_fraction: Ratio, right_fraction: Ratio) -> bool:
    """Absolute span fraction at the panel centre."""
    return twist_fraction == ((left_fraction + right_fraction) ** 2) ** 0.5


@equation
def panel_twist(twist: PanelRatio, root_twist: Ratio, tip_twist: Ratio, twist_fraction: PanelRatio) -> bool:
    """Linearly interpolate twist using absolute normalized span."""
    return twist == root_twist + (tip_twist - root_twist) * twist_fraction


PANEL_MODEL = Implementation(
    SurfacePanels,
    [
        left_edge_fraction,
        right_edge_fraction,
        left_position,
        right_position,
        centre_position,
        panel_width,
        moment_arm,
        spanwise_twist_fraction,
        panel_twist,
    ],
)


def _validate_surface(surface: RectangularSurface, panels: int) -> None:
    values = tuple(
        value.convert_to(meter).value for value in (surface.span, surface.chord, surface.leading_edge_x, surface.height)
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
        return SurfacePanels(
            span=surface.span,
            bound_x=surface.bound_x,
            target_x=surface.target_x,
            height=surface.height,
            cg_x=cg_x,
            root_twist=surface.root_twist,
            tip_twist=surface.tip_twist,
            panel_index=tensor(np.arange(panels), over=PANEL, unit=fraction),
            panel_count=panels * fraction,
        )


@system
class CartesianPoints(metaclass=Interface):
    """Pack dimensional coordinates on a supplied Cartesian axis."""

    x: Length = field()
    y: PanelLength = field()
    z: Length = field()
    x_mask: PanelRatio = field()
    y_mask: PanelRatio = field()
    z_mask: PanelRatio = field()
    points: PanelLength = field()


@equation
def points(  # noqa: PLR0913, PLR0917 - named coordinate components.
    points: PanelLength,
    x: Length,
    y: PanelLength,
    z: Length,
    x_mask: PanelRatio,
    y_mask: PanelRatio,
    z_mask: PanelRatio,
) -> bool:
    """Pack coordinates by projecting onto the Cartesian basis."""
    return points == x_mask * x + y_mask * y + z_mask * z


POINTS_MODEL = Implementation(CartesianPoints, [points])


@system
class HorseshoeEndpoints(metaclass=Interface):
    """Three oriented finite segments on supplied panel and segment axes."""

    left: PanelLength = field()
    right: PanelLength = field()
    wake_length: Length = field()
    streamwise_axis: PanelRatio = field()
    left_segment: PanelRatio = field()
    bound_segment: PanelRatio = field()
    right_segment: PanelRatio = field()
    wake_offset: PanelLength = field()
    left_wake: PanelLength = field()
    right_wake: PanelLength = field()
    start: PanelLength = field()
    end: PanelLength = field()


@equation
def wake_displacement(wake_offset: PanelLength, wake_length: Length, streamwise_axis: PanelRatio) -> bool:
    """Straight downstream wake displacement."""
    return wake_offset == wake_length * streamwise_axis


@equation
def left_wake_position(left_wake: PanelLength, left: PanelLength, wake_offset: PanelLength) -> bool:
    """Downstream endpoint of the left trailing leg."""
    return left_wake == left + wake_offset


@equation
def right_wake_position(right_wake: PanelLength, right: PanelLength, wake_offset: PanelLength) -> bool:
    """Downstream endpoint of the right trailing leg."""
    return right_wake == right + wake_offset


@equation
def start(  # noqa: PLR0913, PLR0917 - named segment endpoints.
    start: PanelLength,
    left_wake: PanelLength,
    left: PanelLength,
    right: PanelLength,
    left_segment: PanelRatio,
    bound_segment: PanelRatio,
    right_segment: PanelRatio,
) -> bool:
    """Project the three start endpoints onto their oriented segment axis."""
    return start == left_segment * left_wake + bound_segment * left + right_segment * right


@equation
def end(  # noqa: PLR0913, PLR0917 - named segment endpoints.
    end: PanelLength,
    left: PanelLength,
    right: PanelLength,
    right_wake: PanelLength,
    left_segment: PanelRatio,
    bound_segment: PanelRatio,
    right_segment: PanelRatio,
) -> bool:
    """Project the three end endpoints onto their oriented segment axis."""
    return end == left_segment * left + bound_segment * right + right_segment * right_wake


ENDPOINTS_MODEL = Implementation(
    HorseshoeEndpoints, [wake_displacement, left_wake_position, right_wake_position, start, end]
)


def _points(x: Quantity, y: QuantityTensor, z: Quantity, panel_dim: Dim) -> QuantityTensor:
    """Bind the requested panel and Cartesian axes at the model boundary."""
    with POINTS_MODEL:
        geometry = CartesianPoints(
            x=x,
            y=tensor(y.convert_to(meter).values, over=panel_dim, unit=meter),
            z=z,
            x_mask=tensor([1, 0, 0], over=CARTESIAN, unit=fraction),
            y_mask=tensor([0, 1, 0], over=CARTESIAN, unit=fraction),
            z_mask=tensor([0, 0, 1], over=CARTESIAN, unit=fraction),
        )
        return geometry.points


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
    checked_cross_squared: FourthPowerLength = field()
    norm1_squared: PanelRatio = field()
    norm2_squared: PanelRatio = field()
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
    return cross_squared == cross_x * cross_x + cross_y * cross_y + cross_z * cross_z


@equation
def first_distance_squared(norm1_squared: PanelRatio, r1: PanelLength) -> bool:
    """Squared Euclidean distance normalised by the reference length."""
    normalised = r1 / _LENGTH_SCALE
    return norm1_squared == (normalised * normalised).sum(CARTESIAN)


@equation
def first_distance(norm1: Length, norm1_squared: Ratio) -> bool:
    """Recover the positive endpoint distance pointwise."""
    return norm1 == _LENGTH_SCALE * norm1_squared**0.5


@equation
def second_distance_squared(norm2_squared: PanelRatio, r2: PanelLength) -> bool:
    """Squared Euclidean distance normalised by the reference length."""
    normalised = r2 / _LENGTH_SCALE
    return norm2_squared == (normalised * normalised).sum(CARTESIAN)


@equation
def second_distance(norm2: Length, norm2_squared: Ratio) -> bool:
    """Recover the positive endpoint distance pointwise."""
    return norm2 == _LENGTH_SCALE * norm2_squared**0.5


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


@transform
def checked_cross_squared(
    cross_squared: FourthPowerLength, norm1: PanelLength, norm2: PanelLength
) -> FourthPowerLength:
    """Reject singular vortex geometry before evaluating the Biot-Savart denominator."""
    from typing import cast  # noqa: PLC0415 - backend runtime dependency.

    import numpy as np  # noqa: PLC0415 - backend runtime dependency.

    # The backend supplies coherent SI magnitudes.
    tolerance = 1e-12
    if (
        np.any(cast("np.ndarray", cross_squared) <= tolerance**4)
        or np.any(cast("np.ndarray", norm1) <= tolerance)
        or np.any(cast("np.ndarray", norm2) <= tolerance)
    ):
        msg = "Vortex evaluation point lies on or too near a filament."
        raise ValueError(msg)
    return cross_squared


@equation
def biot_savart(
    segment_influence: Influence,
    cross_z: SquaredLength,
    endpoint_factor: PanelLength,
    checked_cross_squared: FourthPowerLength,
) -> bool:
    """Vertical finite-segment velocity per unit circulation."""
    return segment_influence == cross_z * endpoint_factor / (4.0 * math.pi * checked_cross_squared)


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
        first_distance_squared,
        second_distance_squared,
        first_distance,
        second_distance,
        endpoint_projection,
        checked_cross_squared,
        biot_savart,
        integrate_segments,
    ],
)


def evaluate_vortices(point: QuantityTensor, start: QuantityTensor, end: QuantityTensor) -> QuantityTensor:
    """Evaluate the model, rejecting singular geometries before division."""
    with VORTEX_MODEL:
        vortex = VortexInfluence(point=point, start=start, end=end)
        return vortex.influence


def influence_block(
    targets: SurfacePanels,
    sources: SurfacePanels,
    *,
    wake_length: Quantity,
    target_dim: Dim,
    source_dim: Dim,
) -> QuantityTensor:
    """Evaluate every target/source interaction using named Bilt tensor axes."""
    with PANEL_MODEL:
        left = _points(sources.bound_x, sources.left_y, sources.height, source_dim)
        right = _points(sources.bound_x, sources.right_y, sources.height, source_dim)
        collocation = _points(targets.target_x, targets.centre_y, targets.height, target_dim)

    with ENDPOINTS_MODEL:
        endpoints = HorseshoeEndpoints(
            left=left,
            right=right,
            wake_length=wake_length,
            streamwise_axis=tensor([1, 0, 0], over=CARTESIAN, unit=fraction),
            left_segment=tensor([1, 0, 0], over=SEGMENT, unit=fraction),
            bound_segment=tensor([0, 1, 0], over=SEGMENT, unit=fraction),
            right_segment=tensor([0, 0, 1], over=SEGMENT, unit=fraction),
        )
        result = evaluate_vortices(collocation, endpoints.start, endpoints.end)
    if result.domain.dims == (target_dim, source_dim):
        return result
    order = tuple(result.domain.dims.index(axis) for axis in (target_dim, source_dim))
    return tensor(result.values.transpose(order), over=(target_dim, source_dim), unit=meter**-1)


class InfluenceInputs(TypedDict):
    """Named dimensional inputs at the numerical kernel / trim model boundary."""

    wing_from_wing: QuantityTensor
    wing_from_tail: QuantityTensor
    tail_from_wing: QuantityTensor
    tail_from_tail: QuantityTensor
    wing_width: QuantityTensor
    tail_width: QuantityTensor
    wing_arm: Quantity
    tail_arm: Quantity
    wing_twist: QuantityTensor
    tail_twist: QuantityTensor


def calculate_influences(geometry: AircraftGeometry, *, resolution: PanelResolution | None = None) -> InfluenceInputs:
    """Evaluate the geometry and supply named influence and load tensors."""
    resolution = resolution if resolution is not None else default_resolution()
    validate_panel_count(resolution.wing)
    validate_panel_count(resolution.tail)
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
            "wing_from_wing": influence_block(
                wing, wing, wake_length=wake, target_dim=WING_TARGET, source_dim=WING_SOURCE
            ),
            "wing_from_tail": influence_block(
                wing, tail, wake_length=wake, target_dim=WING_TARGET, source_dim=TAIL_SOURCE
            ),
            "tail_from_wing": influence_block(
                tail, wing, wake_length=wake, target_dim=TAIL_TARGET, source_dim=WING_SOURCE
            ),
            "tail_from_tail": influence_block(
                tail, tail, wake_length=wake, target_dim=TAIL_TARGET, source_dim=TAIL_SOURCE
            ),
            "wing_width": tensor(wing.width.convert_to(meter).values, over=WING_SOURCE, unit=meter),
            "tail_width": tensor(tail.width.convert_to(meter).values, over=TAIL_SOURCE, unit=meter),
            "wing_arm": wing.arm,
            "tail_arm": tail.arm,
            "wing_twist": tensor(wing.twist.convert_to(radian).values, over=WING_TARGET, unit=radian),
            "tail_twist": tensor(tail.twist.convert_to(radian).values, over=TAIL_TARGET, unit=radian),
        }
