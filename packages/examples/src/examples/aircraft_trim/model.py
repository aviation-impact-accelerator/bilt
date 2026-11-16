"""Acausal wing-and-tail spanwise trim with four independent panel circulations.

Two wing and two tail circulation unknowns are coupled by four prescribed
source-to-target influence blocks. This is not a geometry-derived VLM solver.
"""

from typing import Annotated

from bilt import Domain, Implementation, Interface, Quantity, QuantityTensor, dim, equation, field, system
from bilt.units.dimensions import DIMENSIONLESS, FORCE, LENGTH, MASS, TIME

CASE = dim("case")
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
type WingArm = Annotated[QuantityTensor, LENGTH, Domain((WING_SOURCE,))]
type TailArm = Annotated[QuantityTensor, LENGTH, Domain((TAIL_SOURCE,))]
type WingTwist = Annotated[QuantityTensor, DIMENSIONLESS, Domain((WING_TARGET,))]
type TailTwist = Annotated[QuantityTensor, DIMENSIONLESS, Domain((TAIL_TARGET,))]
type TailControl = Annotated[QuantityTensor, DIMENSIONLESS, Domain((TAIL_TARGET,))]
type Density = Annotated[Quantity, MASS / LENGTH**3]
type Speed = Annotated[Quantity, LENGTH / TIME]
type Angle = Annotated[Quantity | QuantityTensor, DIMENSIONLESS, Domain.scalar()]
type Lift = Annotated[Quantity, FORCE]
type Moment = Annotated[Quantity, FORCE * LENGTH]


@system
class WingTailTrim(metaclass=Interface):
    """Four lifting panels coupled aerodynamically and by longitudinal trim."""

    wing_from_wing: WingInfluence = field()
    wing_from_tail: WingTailInfluence = field()
    tail_from_wing: TailWingInfluence = field()
    tail_from_tail: TailInfluence = field()

    wing_width: WingWidth = field()
    tail_width: TailWidth = field()
    wing_arm: WingArm = field()
    tail_arm: TailArm = field()
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
def weight_balance(  # noqa: PLR0913, PLR0917 - Bilt names the physical equation ports.
    wing_circulation: WingCirculation,
    tail_circulation: TailCirculation,
    wing_width: WingWidth,
    tail_width: TailWidth,
    density: Density,
    speed: Speed,
    mass: Annotated[Quantity, MASS],
    gravity: Annotated[Quantity, LENGTH / TIME**2],
) -> bool:
    """Kutta-Joukowski wing + tail lift balances aircraft weight."""
    return (
        density
        * speed
        * ((wing_circulation * wing_width).sum(WING_SOURCE) + (tail_circulation * tail_width).sum(TAIL_SOURCE))
        == mass * gravity
    )


@equation
def pitch_balance(  # noqa: PLR0913, PLR0917 - Bilt names the physical equation ports.
    wing_circulation: WingCirculation,
    tail_circulation: TailCirculation,
    wing_width: WingWidth,
    tail_width: TailWidth,
    wing_arm: WingArm,
    tail_arm: TailArm,
    density: Density,
    speed: Speed,
    pitch_moment: Moment,
) -> bool:
    """Total pitching moment about CG, with tail arm negative."""
    return (
        density
        * speed
        * (
            (wing_circulation * wing_width * wing_arm).sum(WING_SOURCE)
            + (tail_circulation * tail_width * tail_arm).sum(TAIL_SOURCE)
        )
        == pitch_moment
    )


@equation
def wing_sectional_lift(
    wing_panel_lift: Annotated[Quantity | QuantityTensor, FORCE, Domain((WING_SOURCE,))],
    wing_circulation: WingCirculation,
    wing_width: WingWidth,
    density: Density,
    speed: Speed,
) -> bool:
    """Lift on each independent wing panel."""
    return wing_panel_lift == density * speed * wing_circulation * wing_width


@equation
def tail_sectional_lift(
    tail_panel_lift: Annotated[Quantity | QuantityTensor, FORCE, Domain((TAIL_SOURCE,))],
    tail_circulation: TailCirculation,
    tail_width: TailWidth,
    density: Density,
    speed: Speed,
) -> bool:
    """Lift on each independent tail panel."""
    return tail_panel_lift == density * speed * tail_circulation * tail_width


@equation
def integrate_lift(  # noqa: PLR0913, PLR0917 - Bilt names the physical equation ports.
    lift: Lift,
    wing_circulation: WingCirculation,
    tail_circulation: TailCirculation,
    wing_width: WingWidth,
    tail_width: TailWidth,
    density: Density,
    speed: Speed,
) -> bool:
    """Evaluate total lift after solving the coupled aerodynamic block."""
    return lift == density * speed * (
        (wing_circulation * wing_width).sum(WING_SOURCE) + (tail_circulation * tail_width).sum(TAIL_SOURCE)
    )


MODEL = Implementation(
    WingTailTrim,
    [
        wing_boundary,
        tail_boundary,
        weight_balance,
        pitch_balance,
        wing_sectional_lift,
        tail_sectional_lift,
        integrate_lift,
    ],
)
