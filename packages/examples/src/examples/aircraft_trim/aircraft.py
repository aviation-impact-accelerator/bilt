"""Composed aircraft model: geometry, horseshoe influences and acausal trim."""

from typing import Annotated

from bilt import Domain, Implementation, Interface, Quantity, QuantityTensor, equation, field, system, transform
from bilt.units.dimensions import DIMENSIONLESS, LENGTH, MASS, TIME

from .geometry import (
    ENDPOINTS_MODEL,
    PANEL_MODEL,
    SURFACE_MODEL,
    VORTEX_MODEL,
    Area,
    HorseshoeEndpoints,
    Length,
    PanelLength,
    PanelRatio,
    Ratio,
    RectangularSurface,
    SurfacePanels,
    VortexInfluence,
)
from .model import (
    TAIL_SOURCE,
    WING_SOURCE,
    Density,
    Lift,
    Moment,
    MomentArm,
    Speed,
    TailControl,
    TailInfluence,
    TailPanelLift,
    TailTwist,
    TailWidth,
    TailWingInfluence,
    WingInfluence,
    WingPanelLift,
    WingTailInfluence,
    WingTwist,
    WingWidth,
    integrate_lift,
    pitch_balance,
    tail_boundary,
    tail_sectional_lift,
    weight_balance,
    wing_boundary,
    wing_sectional_lift,
)


@system
class SurfaceLattice(metaclass=Interface):
    """One lifting surface, with separate source and target panel axes."""

    span: Length = field()
    chord: Length = field()
    leading_edge_x: Length = field()
    height: Length = field()
    root_twist: Ratio = field()
    tip_twist: Ratio = field()
    validated_span: Length = field()
    cg_x: Length = field()
    wake_length: Length = field()
    panel_count: Ratio = field()
    source_index: PanelRatio = field()
    target_index: PanelRatio = field()
    x_mask: PanelRatio = field()
    y_mask: PanelRatio = field()
    z_mask: PanelRatio = field()
    left_segment: PanelRatio = field()
    bound_segment: PanelRatio = field()
    right_segment: PanelRatio = field()
    area: Area = field()
    width: PanelLength = field()
    arm: Length = field()
    twist: PanelRatio = field()
    point: PanelLength = field()
    start: PanelLength = field()
    end: PanelLength = field()


@transform
def validated_span(  # noqa: PLR0913, PLR0917 - physical geometry inputs.
    span: Length,
    chord: Length,
    leading_edge_x: Length,
    height: Length,
    root_twist: Ratio,
    tip_twist: Ratio,
    panel_count: Ratio,
) -> Length:
    """Validate a physical surface before constructing its lattice."""
    from typing import cast  # noqa: PLC0415 - backend runtime dependency.

    import math  # noqa: PLC0415 - backend runtime dependency.

    span_value = cast("float", span)
    chord_value = cast("float", chord)
    count = cast("float", panel_count)
    coordinates = (span_value, chord_value, cast("float", leading_edge_x), cast("float", height))
    if not all(math.isfinite(value) for value in coordinates):
        msg = "Lifting-surface geometry must contain finite coordinates."
        raise ValueError(msg)
    if span_value <= 0 or chord_value <= 0 or not math.isfinite(count) or count < 1 or count % 1 != 0:
        msg = "Lifting-surface span, chord and panel count must be positive."
        raise ValueError(msg)
    if not all(math.isfinite(value) for value in (cast("float", root_twist), cast("float", tip_twist))):
        msg = "Lifting-surface twist must be finite."
        raise ValueError(msg)
    return span


@equation
def surface_span(validated_span: Length, surface: RectangularSurface) -> bool:
    """Connect validated span to surface."""
    return validated_span == surface.span


@equation
def surface_chord(chord: Length, surface: RectangularSurface) -> bool:
    """Connect chord to surface."""
    return chord == surface.chord


@equation
def surface_leading_edge_x(leading_edge_x: Length, surface: RectangularSurface) -> bool:
    """Connect leading edge x to surface."""
    return leading_edge_x == surface.leading_edge_x


@equation
def surface_height(height: Length, surface: RectangularSurface) -> bool:
    """Connect height to surface."""
    return height == surface.height


@equation
def surface_root_twist(root_twist: Ratio, surface: RectangularSurface) -> bool:
    """Connect root twist to surface."""
    return root_twist == surface.root_twist


@equation
def surface_tip_twist(tip_twist: Ratio, surface: RectangularSurface) -> bool:
    """Connect tip twist to surface."""
    return tip_twist == surface.tip_twist


@equation
def surface_area_connection(area: Area, surface: RectangularSurface) -> bool:
    """Connect area to surface."""
    return area == surface.area


@equation
def source_panel_validated_span(validated_span: Length, source_panel: SurfacePanels) -> bool:
    """Connect validated span to source panel."""
    return validated_span == source_panel.span


@equation
def source_panel_panel_count(panel_count: Ratio, source_panel: SurfacePanels) -> bool:
    """Connect panel count to source panel."""
    return panel_count == source_panel.panel_count


@equation
def source_panel_source_index(source_index: PanelRatio, source_panel: SurfacePanels) -> bool:
    """Connect source index to source panel."""
    return source_index == source_panel.panel_index


@equation
def target_panel_validated_span(validated_span: Length, target_panel: SurfacePanels) -> bool:
    """Connect validated span to target panel."""
    return validated_span == target_panel.span


@equation
def target_panel_panel_count(panel_count: Ratio, target_panel: SurfacePanels) -> bool:
    """Connect panel count to target panel."""
    return panel_count == target_panel.panel_count


@equation
def target_panel_target_index(target_index: PanelRatio, target_panel: SurfacePanels) -> bool:
    """Connect target index to target panel."""
    return target_index == target_panel.panel_index


@equation
def source_bound_position(source_panel: SurfacePanels, surface: RectangularSurface) -> bool:
    """Connect source panel bound x to surface."""
    return source_panel.bound_x == surface.bound_x


@equation
def target_collocation_position(target_panel: SurfacePanels, surface: RectangularSurface) -> bool:
    """Connect target panel target x to surface."""
    return target_panel.target_x == surface.target_x


@equation
def source_cg(cg_x: Length, source_panel: SurfacePanels) -> bool:
    """Connect cg x to source panel."""
    return cg_x == source_panel.cg_x


@equation
def target_root_twist(root_twist: Ratio, target_panel: SurfacePanels) -> bool:
    """Connect root twist to target panel."""
    return root_twist == target_panel.root_twist


@equation
def target_tip_twist(tip_twist: Ratio, target_panel: SurfacePanels) -> bool:
    """Connect tip twist to target panel."""
    return tip_twist == target_panel.tip_twist


@equation
def expose_width(width: PanelLength, source_panel: SurfacePanels) -> bool:
    """Connect width to source panel."""
    return width == source_panel.width


@equation
def expose_arm(arm: Length, source_panel: SurfacePanels) -> bool:
    """Connect arm to source panel."""
    return arm == source_panel.arm


@equation
def expose_twist(twist: PanelRatio, target_panel: SurfacePanels) -> bool:
    """Connect twist to target panel."""
    return twist == target_panel.twist


@equation
def pack_left(  # noqa: PLR0913, PLR0917 - named coordinate components.
    source_panel: SurfacePanels,
    endpoints: HorseshoeEndpoints,
    height: Length,
    x_mask: PanelRatio,
    y_mask: PanelRatio,
    z_mask: PanelRatio,
) -> bool:
    """Pack the left bound-vortex endpoint in the Cartesian basis."""
    return endpoints.left == x_mask * source_panel.bound_x + y_mask * source_panel.left_y + z_mask * height


@equation
def pack_right(  # noqa: PLR0913, PLR0917 - named coordinate components.
    source_panel: SurfacePanels,
    endpoints: HorseshoeEndpoints,
    height: Length,
    x_mask: PanelRatio,
    y_mask: PanelRatio,
    z_mask: PanelRatio,
) -> bool:
    """Pack the right bound-vortex endpoint in the Cartesian basis."""
    return endpoints.right == x_mask * source_panel.bound_x + y_mask * source_panel.right_y + z_mask * height


@equation
def pack_collocation(  # noqa: PLR0913, PLR0917 - named coordinate components.
    point: PanelLength,
    target_panel: SurfacePanels,
    height: Length,
    x_mask: PanelRatio,
    y_mask: PanelRatio,
    z_mask: PanelRatio,
) -> bool:
    """Pack the target collocation point in the Cartesian basis."""
    return point == x_mask * target_panel.target_x + y_mask * target_panel.centre_y + z_mask * height


@equation
def endpoints_wake_length(wake_length: Length, endpoints: HorseshoeEndpoints) -> bool:
    """Connect wake length to endpoints."""
    return wake_length == endpoints.wake_length


@equation
def endpoints_x_mask(x_mask: PanelRatio, endpoints: HorseshoeEndpoints) -> bool:
    """Connect x mask to endpoints."""
    return x_mask == endpoints.streamwise_axis


@equation
def endpoints_left_segment(left_segment: PanelRatio, endpoints: HorseshoeEndpoints) -> bool:
    """Connect left segment to endpoints."""
    return left_segment == endpoints.left_segment


@equation
def endpoints_bound_segment(bound_segment: PanelRatio, endpoints: HorseshoeEndpoints) -> bool:
    """Connect bound segment to endpoints."""
    return bound_segment == endpoints.bound_segment


@equation
def endpoints_right_segment(right_segment: PanelRatio, endpoints: HorseshoeEndpoints) -> bool:
    """Connect right segment to endpoints."""
    return right_segment == endpoints.right_segment


@equation
def endpoints_start(start: PanelLength, endpoints: HorseshoeEndpoints) -> bool:
    """Connect start to endpoints."""
    return start == endpoints.start


@equation
def endpoints_end(end: PanelLength, endpoints: HorseshoeEndpoints) -> bool:
    """Connect end to endpoints."""
    return end == endpoints.end


SURFACE_LATTICE_MODEL = Implementation(
    SurfaceLattice,
    [
        validated_span,
        surface_span,
        surface_chord,
        surface_leading_edge_x,
        surface_height,
        surface_root_twist,
        surface_tip_twist,
        surface_area_connection,
        source_panel_validated_span,
        source_panel_panel_count,
        source_panel_source_index,
        target_panel_validated_span,
        target_panel_panel_count,
        target_panel_target_index,
        source_bound_position,
        target_collocation_position,
        source_cg,
        target_root_twist,
        target_tip_twist,
        expose_width,
        expose_arm,
        expose_twist,
        pack_left,
        pack_right,
        pack_collocation,
        endpoints_wake_length,
        endpoints_x_mask,
        endpoints_left_segment,
        endpoints_bound_segment,
        endpoints_right_segment,
        endpoints_start,
        endpoints_end,
    ],
    internal_implementations={
        "surface": SURFACE_MODEL,
        "source_panel": PANEL_MODEL,
        "target_panel": PANEL_MODEL,
        "endpoints": ENDPOINTS_MODEL,
    },
)


@system
class Aircraft(metaclass=Interface):
    """Physical aircraft and flight conditions with calculated trim outputs."""

    wing_span: Length = field()
    wing_chord: Length = field()
    wing_leading_edge_x: Length = field()
    wing_height: Length = field()
    wing_root_twist: Ratio = field()
    wing_tip_twist: Ratio = field()
    tail_span: Length = field()
    tail_chord: Length = field()
    tail_leading_edge_x: Length = field()
    tail_height: Length = field()
    tail_root_twist: Ratio = field()
    tail_tip_twist: Ratio = field()
    wing_area: Area = field()
    tail_area: Area = field()
    cg_x: Length = field()
    wake_length: Length = field()
    validated_wake_length: Length = field()
    wing_panel_count: Ratio = field()
    wing_source_index: PanelRatio = field()
    wing_target_index: PanelRatio = field()
    tail_panel_count: Ratio = field()
    tail_source_index: PanelRatio = field()
    tail_target_index: PanelRatio = field()
    x_mask: PanelRatio = field()
    y_mask: PanelRatio = field()
    z_mask: PanelRatio = field()
    left_segment: PanelRatio = field()
    bound_segment: PanelRatio = field()
    right_segment: PanelRatio = field()

    wing_from_wing: WingInfluence = field()
    wing_from_tail: WingTailInfluence = field()
    tail_from_wing: TailWingInfluence = field()
    tail_from_tail: TailInfluence = field()
    wing_width: WingWidth = field()
    tail_width: TailWidth = field()
    wing_arm: MomentArm = field()
    tail_arm: MomentArm = field()
    wing_twist: WingTwist = field()
    tail_twist: TailTwist = field()
    tail_control: TailControl = field()
    density: Density = field()
    speed: Speed = field()
    gravity: Annotated[Quantity, LENGTH / TIME**2] = field()
    mass: Annotated[Quantity, MASS] = field()
    alpha: Annotated[Quantity, DIMENSIONLESS] = field()
    elevator: Annotated[Quantity, DIMENSIONLESS] = field()
    wing_circulation: Annotated[QuantityTensor, LENGTH**2 / TIME, Domain((WING_SOURCE,))] = field()
    tail_circulation: Annotated[QuantityTensor, LENGTH**2 / TIME, Domain((TAIL_SOURCE,))] = field()
    wing_panel_lift: WingPanelLift = field()
    tail_panel_lift: TailPanelLift = field()
    lift: Lift = field()
    pitch_moment: Moment = field()


@transform
def validated_wake_length(
    cg_x: Length, wake_length: Length, wing_leading_edge_x: Length, tail_leading_edge_x: Length
) -> Length:
    """Validate wake length, CG and downstream tail placement."""
    import math  # noqa: PLC0415 - backend runtime dependency.
    from typing import cast  # noqa: PLC0415 - backend runtime dependency.

    wake = cast("float", wake_length)
    if not math.isfinite(cast("float", cg_x)) or not math.isfinite(wake) or wake <= 0:
        msg = "CG coordinate must be finite and wake length positive."
        raise ValueError(msg)
    if cast("float", tail_leading_edge_x) <= cast("float", wing_leading_edge_x):
        msg = "The horizontal tail must be downstream of the wing."
        raise ValueError(msg)
    return wake_length


@equation
def wing_span_connection(wing_span: Length, wing: SurfaceLattice) -> bool:
    """Connect wing span to wing."""
    return wing_span == wing.span


@equation
def wing_chord_connection(wing_chord: Length, wing: SurfaceLattice) -> bool:
    """Connect wing chord to wing."""
    return wing_chord == wing.chord


@equation
def wing_leading_edge_x_connection(wing_leading_edge_x: Length, wing: SurfaceLattice) -> bool:
    """Connect wing leading edge x to wing."""
    return wing_leading_edge_x == wing.leading_edge_x


@equation
def wing_height_connection(wing_height: Length, wing: SurfaceLattice) -> bool:
    """Connect wing height to wing."""
    return wing_height == wing.height


@equation
def wing_root_twist_connection(wing_root_twist: Ratio, wing: SurfaceLattice) -> bool:
    """Connect wing root twist to wing."""
    return wing_root_twist == wing.root_twist


@equation
def wing_tip_twist_connection(wing_tip_twist: Ratio, wing: SurfaceLattice) -> bool:
    """Connect wing tip twist to wing."""
    return wing_tip_twist == wing.tip_twist


@equation
def wing_area_connection(wing_area: Area, wing: SurfaceLattice) -> bool:
    """Connect wing area to wing."""
    return wing_area == wing.area


@equation
def wing_panel_count_connection(wing_panel_count: Ratio, wing: SurfaceLattice) -> bool:
    """Connect wing panel count to wing."""
    return wing_panel_count == wing.panel_count


@equation
def wing_source_index_connection(wing_source_index: PanelRatio, wing: SurfaceLattice) -> bool:
    """Connect wing source index to wing."""
    return wing_source_index == wing.source_index


@equation
def wing_target_index_connection(wing_target_index: PanelRatio, wing: SurfaceLattice) -> bool:
    """Connect wing target index to wing."""
    return wing_target_index == wing.target_index


@equation
def wing_cg_x_connection(cg_x: Length, wing: SurfaceLattice) -> bool:
    """Connect cg x to wing."""
    return cg_x == wing.cg_x


@equation
def wing_validated_wake_length_connection(validated_wake_length: Length, wing: SurfaceLattice) -> bool:
    """Connect validated wake length to wing."""
    return validated_wake_length == wing.wake_length


@equation
def wing_x_mask_connection(x_mask: PanelRatio, wing: SurfaceLattice) -> bool:
    """Connect x mask to wing."""
    return x_mask == wing.x_mask


@equation
def wing_y_mask_connection(y_mask: PanelRatio, wing: SurfaceLattice) -> bool:
    """Connect y mask to wing."""
    return y_mask == wing.y_mask


@equation
def wing_z_mask_connection(z_mask: PanelRatio, wing: SurfaceLattice) -> bool:
    """Connect z mask to wing."""
    return z_mask == wing.z_mask


@equation
def wing_left_segment_connection(left_segment: PanelRatio, wing: SurfaceLattice) -> bool:
    """Connect left segment to wing."""
    return left_segment == wing.left_segment


@equation
def wing_bound_segment_connection(bound_segment: PanelRatio, wing: SurfaceLattice) -> bool:
    """Connect bound segment to wing."""
    return bound_segment == wing.bound_segment


@equation
def wing_right_segment_connection(right_segment: PanelRatio, wing: SurfaceLattice) -> bool:
    """Connect right segment to wing."""
    return right_segment == wing.right_segment


@equation
def wing_width_connection(wing_width: WingWidth, wing: SurfaceLattice) -> bool:
    """Connect wing width to wing."""
    return wing_width == wing.width


@equation
def wing_arm_connection(wing_arm: MomentArm, wing: SurfaceLattice) -> bool:
    """Connect wing arm to wing."""
    return wing_arm == wing.arm


@equation
def wing_twist_connection(wing_twist: WingTwist, wing: SurfaceLattice) -> bool:
    """Connect wing twist to wing."""
    return wing_twist == wing.twist


@equation
def tail_span_connection(tail_span: Length, tail: SurfaceLattice) -> bool:
    """Connect tail span to tail."""
    return tail_span == tail.span


@equation
def tail_chord_connection(tail_chord: Length, tail: SurfaceLattice) -> bool:
    """Connect tail chord to tail."""
    return tail_chord == tail.chord


@equation
def tail_leading_edge_x_connection(tail_leading_edge_x: Length, tail: SurfaceLattice) -> bool:
    """Connect tail leading edge x to tail."""
    return tail_leading_edge_x == tail.leading_edge_x


@equation
def tail_height_connection(tail_height: Length, tail: SurfaceLattice) -> bool:
    """Connect tail height to tail."""
    return tail_height == tail.height


@equation
def tail_root_twist_connection(tail_root_twist: Ratio, tail: SurfaceLattice) -> bool:
    """Connect tail root twist to tail."""
    return tail_root_twist == tail.root_twist


@equation
def tail_tip_twist_connection(tail_tip_twist: Ratio, tail: SurfaceLattice) -> bool:
    """Connect tail tip twist to tail."""
    return tail_tip_twist == tail.tip_twist


@equation
def tail_area_connection(tail_area: Area, tail: SurfaceLattice) -> bool:
    """Connect tail area to tail."""
    return tail_area == tail.area


@equation
def tail_panel_count_connection(tail_panel_count: Ratio, tail: SurfaceLattice) -> bool:
    """Connect tail panel count to tail."""
    return tail_panel_count == tail.panel_count


@equation
def tail_source_index_connection(tail_source_index: PanelRatio, tail: SurfaceLattice) -> bool:
    """Connect tail source index to tail."""
    return tail_source_index == tail.source_index


@equation
def tail_target_index_connection(tail_target_index: PanelRatio, tail: SurfaceLattice) -> bool:
    """Connect tail target index to tail."""
    return tail_target_index == tail.target_index


@equation
def tail_cg_x_connection(cg_x: Length, tail: SurfaceLattice) -> bool:
    """Connect cg x to tail."""
    return cg_x == tail.cg_x


@equation
def tail_validated_wake_length_connection(validated_wake_length: Length, tail: SurfaceLattice) -> bool:
    """Connect validated wake length to tail."""
    return validated_wake_length == tail.wake_length


@equation
def tail_x_mask_connection(x_mask: PanelRatio, tail: SurfaceLattice) -> bool:
    """Connect x mask to tail."""
    return x_mask == tail.x_mask


@equation
def tail_y_mask_connection(y_mask: PanelRatio, tail: SurfaceLattice) -> bool:
    """Connect y mask to tail."""
    return y_mask == tail.y_mask


@equation
def tail_z_mask_connection(z_mask: PanelRatio, tail: SurfaceLattice) -> bool:
    """Connect z mask to tail."""
    return z_mask == tail.z_mask


@equation
def tail_left_segment_connection(left_segment: PanelRatio, tail: SurfaceLattice) -> bool:
    """Connect left segment to tail."""
    return left_segment == tail.left_segment


@equation
def tail_bound_segment_connection(bound_segment: PanelRatio, tail: SurfaceLattice) -> bool:
    """Connect bound segment to tail."""
    return bound_segment == tail.bound_segment


@equation
def tail_right_segment_connection(right_segment: PanelRatio, tail: SurfaceLattice) -> bool:
    """Connect right segment to tail."""
    return right_segment == tail.right_segment


@equation
def tail_width_connection(tail_width: TailWidth, tail: SurfaceLattice) -> bool:
    """Connect tail width to tail."""
    return tail_width == tail.width


@equation
def tail_arm_connection(tail_arm: MomentArm, tail: SurfaceLattice) -> bool:
    """Connect tail arm to tail."""
    return tail_arm == tail.arm


@equation
def tail_twist_connection(tail_twist: TailTwist, tail: SurfaceLattice) -> bool:
    """Connect tail twist to tail."""
    return tail_twist == tail.twist


@equation
def wing_from_wing_point(wing_from_wing_vortex: VortexInfluence, wing: SurfaceLattice) -> bool:
    """Connect wing from wing vortex point to wing."""
    return wing_from_wing_vortex.point == wing.point


@equation
def wing_from_wing_start(wing_from_wing_vortex: VortexInfluence, wing: SurfaceLattice) -> bool:
    """Connect wing from wing vortex start to wing."""
    return wing_from_wing_vortex.start == wing.start


@equation
def wing_from_wing_end(wing_from_wing_vortex: VortexInfluence, wing: SurfaceLattice) -> bool:
    """Connect wing from wing vortex end to wing."""
    return wing_from_wing_vortex.end == wing.end


@equation
def wing_from_wing_connection(wing_from_wing: WingInfluence, wing_from_wing_vortex: VortexInfluence) -> bool:
    """Connect wing from wing to wing from wing vortex."""
    return wing_from_wing == wing_from_wing_vortex.influence


@equation
def wing_from_tail_point(wing_from_tail_vortex: VortexInfluence, wing: SurfaceLattice) -> bool:
    """Connect wing from tail vortex point to wing."""
    return wing_from_tail_vortex.point == wing.point


@equation
def wing_from_tail_start(wing_from_tail_vortex: VortexInfluence, tail: SurfaceLattice) -> bool:
    """Connect wing from tail vortex start to tail."""
    return wing_from_tail_vortex.start == tail.start


@equation
def wing_from_tail_end(wing_from_tail_vortex: VortexInfluence, tail: SurfaceLattice) -> bool:
    """Connect wing from tail vortex end to tail."""
    return wing_from_tail_vortex.end == tail.end


@equation
def wing_from_tail_connection(wing_from_tail: WingTailInfluence, wing_from_tail_vortex: VortexInfluence) -> bool:
    """Connect wing from tail to wing from tail vortex."""
    return wing_from_tail == wing_from_tail_vortex.influence


@equation
def tail_from_wing_point(tail_from_wing_vortex: VortexInfluence, tail: SurfaceLattice) -> bool:
    """Connect tail from wing vortex point to tail."""
    return tail_from_wing_vortex.point == tail.point


@equation
def tail_from_wing_start(tail_from_wing_vortex: VortexInfluence, wing: SurfaceLattice) -> bool:
    """Connect tail from wing vortex start to wing."""
    return tail_from_wing_vortex.start == wing.start


@equation
def tail_from_wing_end(tail_from_wing_vortex: VortexInfluence, wing: SurfaceLattice) -> bool:
    """Connect tail from wing vortex end to wing."""
    return tail_from_wing_vortex.end == wing.end


@equation
def tail_from_wing_connection(tail_from_wing: TailWingInfluence, tail_from_wing_vortex: VortexInfluence) -> bool:
    """Connect tail from wing to tail from wing vortex."""
    return tail_from_wing == tail_from_wing_vortex.influence


@equation
def tail_from_tail_point(tail_from_tail_vortex: VortexInfluence, tail: SurfaceLattice) -> bool:
    """Connect tail from tail vortex point to tail."""
    return tail_from_tail_vortex.point == tail.point


@equation
def tail_from_tail_start(tail_from_tail_vortex: VortexInfluence, tail: SurfaceLattice) -> bool:
    """Connect tail from tail vortex start to tail."""
    return tail_from_tail_vortex.start == tail.start


@equation
def tail_from_tail_end(tail_from_tail_vortex: VortexInfluence, tail: SurfaceLattice) -> bool:
    """Connect tail from tail vortex end to tail."""
    return tail_from_tail_vortex.end == tail.end


@equation
def tail_from_tail_connection(tail_from_tail: TailInfluence, tail_from_tail_vortex: VortexInfluence) -> bool:
    """Connect tail from tail to tail from tail vortex."""
    return tail_from_tail == tail_from_tail_vortex.influence


MODEL = Implementation(
    Aircraft,
    [
        validated_wake_length,
        wing_span_connection,
        wing_chord_connection,
        wing_leading_edge_x_connection,
        wing_height_connection,
        wing_root_twist_connection,
        wing_tip_twist_connection,
        wing_area_connection,
        wing_panel_count_connection,
        wing_source_index_connection,
        wing_target_index_connection,
        wing_cg_x_connection,
        wing_validated_wake_length_connection,
        wing_x_mask_connection,
        wing_y_mask_connection,
        wing_z_mask_connection,
        wing_left_segment_connection,
        wing_bound_segment_connection,
        wing_right_segment_connection,
        wing_width_connection,
        wing_arm_connection,
        wing_twist_connection,
        tail_span_connection,
        tail_chord_connection,
        tail_leading_edge_x_connection,
        tail_height_connection,
        tail_root_twist_connection,
        tail_tip_twist_connection,
        tail_area_connection,
        tail_panel_count_connection,
        tail_source_index_connection,
        tail_target_index_connection,
        tail_cg_x_connection,
        tail_validated_wake_length_connection,
        tail_x_mask_connection,
        tail_y_mask_connection,
        tail_z_mask_connection,
        tail_left_segment_connection,
        tail_bound_segment_connection,
        tail_right_segment_connection,
        tail_width_connection,
        tail_arm_connection,
        tail_twist_connection,
        wing_from_wing_point,
        wing_from_wing_start,
        wing_from_wing_end,
        wing_from_wing_connection,
        wing_from_tail_point,
        wing_from_tail_start,
        wing_from_tail_end,
        wing_from_tail_connection,
        tail_from_wing_point,
        tail_from_wing_start,
        tail_from_wing_end,
        tail_from_wing_connection,
        tail_from_tail_point,
        tail_from_tail_start,
        tail_from_tail_end,
        tail_from_tail_connection,
        wing_boundary,
        tail_boundary,
        wing_sectional_lift,
        tail_sectional_lift,
        integrate_lift,
        weight_balance,
        pitch_balance,
    ],
    internal_implementations={
        "wing": SURFACE_LATTICE_MODEL,
        "tail": SURFACE_LATTICE_MODEL,
        "wing_from_wing_vortex": VORTEX_MODEL,
        "wing_from_tail_vortex": VORTEX_MODEL,
        "tail_from_wing_vortex": VORTEX_MODEL,
        "tail_from_tail_vortex": VORTEX_MODEL,
    },
)
