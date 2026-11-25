"""Acausal wing-and-tail spanwise trim with four independent panel circulations.

Wing and tail panel circulations are coupled by four geometry-derived
horseshoe-vortex influence blocks. This is a spanwise lifting-line-style
model rather than a chordwise vortex-lattice method.
"""

from typing import Annotated

from bilt import Domain, Implementation, Interface, Quantity, QuantityTensor, dim, equation, field, system
from bilt.units.dimensions import DIMENSIONLESS, FORCE, LENGTH, MASS, TIME

WING_SOURCE = dim("wing_source")
TAIL_SOURCE = dim("tail_source")
WING_TARGET = dim("wing_target")
TAIL_TARGET = dim("tail_target")

type WingCirculation = Annotated[Quantity | QuantityTensor, LENGTH**2 / TIME, Domain((WING_SOURCE,))]
type TailCirculation = Annotated[Quantity | QuantityTensor, LENGTH**2 / TIME, Domain((TAIL_SOURCE,))]
type WingInfluence = Annotated[QuantityTensor, LENGTH**-1, Domain((WING_TARGET, WING_SOURCE))]
type WingTailInfluence = Annotated[QuantityTensor, LENGTH**-1, Domain((WING_TARGET, TAIL_SOURCE))]
type TailWingInfluence = Annotated[QuantityTensor, LENGTH**-1, Domain((TAIL_TARGET, WING_SOURCE))]
type TailInfluence = Annotated[QuantityTensor, LENGTH**-1, Domain((TAIL_TARGET, TAIL_SOURCE))]
type WingWidth = Annotated[QuantityTensor, LENGTH, Domain((WING_SOURCE,))]
type TailWidth = Annotated[QuantityTensor, LENGTH, Domain((TAIL_SOURCE,))]
type MomentArm = Annotated[Quantity, LENGTH]
type WingTwist = Annotated[QuantityTensor, DIMENSIONLESS, Domain((WING_TARGET,))]
type TailTwist = Annotated[QuantityTensor, DIMENSIONLESS, Domain((TAIL_TARGET,))]
type TailControl = Annotated[Quantity, DIMENSIONLESS]
type WingPanelLift = Annotated[Quantity | QuantityTensor, FORCE, Domain((WING_SOURCE,))]
type TailPanelLift = Annotated[Quantity | QuantityTensor, FORCE, Domain((TAIL_SOURCE,))]
type Density = Annotated[Quantity, MASS / LENGTH**3]
type Speed = Annotated[Quantity, LENGTH / TIME]
type Angle = Annotated[Quantity | QuantityTensor, DIMENSIONLESS, Domain.scalar()]
type Lift = Annotated[Quantity, FORCE]
type Moment = Annotated[Quantity, FORCE * LENGTH]


@system
class WingTailTrim(metaclass=Interface):
    """Wing and tail source panels coupled by geometric induced velocities."""

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
    wing_panel_lift: Annotated[QuantityTensor, FORCE, Domain((WING_SOURCE,))] = field()
    tail_panel_lift: Annotated[QuantityTensor, FORCE, Domain((TAIL_SOURCE,))] = field()
    lift: Lift = field()
    pitch_moment: Moment = field()


@equation
def wing_boundary(  # noqa: PLR0913, PLR0917 - Bilt names the physical equation ports.
    wing_from_wing: WingInfluence,
    wing_from_tail: WingTailInfluence,
    wing_circulation: WingCirculation,
    tail_circulation: TailCirculation,
    speed: Speed,
    alpha: Angle,
    wing_twist: WingTwist,
) -> bool:
    """Target-panel normal velocity includes both wing and tail sources."""
    return (wing_from_wing * wing_circulation).sum(WING_SOURCE) + (wing_from_tail * tail_circulation).sum(
        TAIL_SOURCE
    ) == -speed * (alpha + wing_twist)


@equation
def tail_boundary(  # noqa: PLR0913, PLR0917 - Bilt names the physical equation ports.
    tail_from_wing: TailWingInfluence,
    tail_from_tail: TailInfluence,
    wing_circulation: WingCirculation,
    tail_circulation: TailCirculation,
    speed: Speed,
    alpha: Angle,
    elevator: Angle,
    tail_twist: TailTwist,
    tail_control: TailControl,
) -> bool:
    """Tail targets see wing and tail sources plus control deflection."""
    return (tail_from_wing * wing_circulation).sum(WING_SOURCE) + (tail_from_tail * tail_circulation).sum(
        TAIL_SOURCE
    ) == -speed * (alpha + tail_twist + tail_control * elevator)


@equation
def weight_balance(
    lift: Lift,
    mass: Annotated[Quantity, MASS],
    gravity: Annotated[Quantity, LENGTH / TIME**2],
) -> bool:
    """Steady level flight requires lift to balance weight."""
    return lift == mass * gravity


@equation
def pitch_balance(
    wing_panel_lift: WingPanelLift,
    tail_panel_lift: TailPanelLift,
    wing_arm: MomentArm,
    tail_arm: MomentArm,
    pitch_moment: Moment,
) -> bool:
    """Total pitching moment about CG, with tail arm negative."""
    return (wing_panel_lift * wing_arm).sum(WING_SOURCE) + (tail_panel_lift * tail_arm).sum(TAIL_SOURCE) == pitch_moment


@equation
def wing_sectional_lift(
    wing_panel_lift: WingPanelLift,
    wing_circulation: WingCirculation,
    wing_width: WingWidth,
    density: Density,
    speed: Speed,
) -> bool:
    """Lift on each independent wing panel."""
    return wing_panel_lift == density * speed * wing_circulation * wing_width


@equation
def tail_sectional_lift(
    tail_panel_lift: TailPanelLift,
    tail_circulation: TailCirculation,
    tail_width: TailWidth,
    density: Density,
    speed: Speed,
) -> bool:
    """Lift on each independent tail panel."""
    return tail_panel_lift == density * speed * tail_circulation * tail_width


@equation
def integrate_lift(
    lift: Lift,
    wing_panel_lift: WingPanelLift,
    tail_panel_lift: TailPanelLift,
) -> bool:
    """Total lift is the sum of the two sectional load distributions."""
    return lift == wing_panel_lift.sum(WING_SOURCE) + tail_panel_lift.sum(TAIL_SOURCE)


MODEL = Implementation(
    WingTailTrim,
    [
        wing_boundary,
        tail_boundary,
        wing_sectional_lift,
        tail_sectional_lift,
        integrate_lift,
        weight_balance,
        pitch_balance,
    ],
)
