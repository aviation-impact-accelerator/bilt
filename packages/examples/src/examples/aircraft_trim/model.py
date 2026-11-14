"""Acausal, linear wing-and-tail longitudinal trim using named panel tensors.

The influence matrix is an illustrative input, not a vortex-lattice calculation.
Angles use the small-angle convention: their coherent magnitudes are radians.
"""

from typing import Annotated

from bilt import Domain, Implementation, Interface, Quantity, QuantityTensor, dim, equation, field, system
from bilt.units.dimensions import DIMENSIONLESS, FORCE, LENGTH, MASS, TIME

CASE = dim("case")
SOURCE_PANEL = dim("source_panel")
TARGET_PANEL = dim("target_panel")


@system
class WingTailTrim(metaclass=Interface):
    """Rigid-wing and tail aerodynamic balance with a linear influence matrix.

    Panel arrays include both wing and tail, in matching source/target order.
    Positive moment arm means lift creates a nose-up pitching moment.
    """

    influence: Annotated[QuantityTensor, LENGTH**-1, Domain((TARGET_PANEL, SOURCE_PANEL))] = field()
    panel_width: Annotated[QuantityTensor, LENGTH, Domain((SOURCE_PANEL,))] = field()
    moment_arm: Annotated[QuantityTensor, LENGTH, Domain((SOURCE_PANEL,))] = field()
    twist: Annotated[QuantityTensor, DIMENSIONLESS, Domain((TARGET_PANEL,))] = field()
    control_effect: Annotated[QuantityTensor, DIMENSIONLESS, Domain((TARGET_PANEL,))] = field()

    density: Annotated[Quantity | QuantityTensor, MASS / LENGTH**3] = field()
    speed: Annotated[Quantity | QuantityTensor, LENGTH / TIME] = field()
    gravity: Annotated[Quantity, LENGTH / TIME**2] = field()
    mass: Annotated[Quantity | QuantityTensor, MASS] = field()

    alpha: Annotated[Quantity | QuantityTensor, DIMENSIONLESS] = field()
    elevator: Annotated[Quantity | QuantityTensor, DIMENSIONLESS] = field()
    circulation: Annotated[QuantityTensor, LENGTH**2 / TIME, Domain((SOURCE_PANEL,))] = field()
    panel_lift: Annotated[QuantityTensor, FORCE, Domain((SOURCE_PANEL,))] = field()
    lift: Annotated[Quantity | QuantityTensor, FORCE] = field()
    pitch_moment: Annotated[Quantity | QuantityTensor, FORCE * LENGTH] = field()


@equation
def no_penetration(
    influence: Annotated[QuantityTensor, LENGTH**-1, Domain((TARGET_PANEL, SOURCE_PANEL))],
    circulation: Annotated[QuantityTensor, LENGTH**2 / TIME, Domain((SOURCE_PANEL,))],
    speed: Annotated[Quantity | QuantityTensor, LENGTH / TIME],
    alpha: Annotated[Quantity | QuantityTensor, DIMENSIONLESS],
    elevator: Annotated[Quantity | QuantityTensor, DIMENSIONLESS],
    twist: Annotated[QuantityTensor, DIMENSIONLESS, Domain((TARGET_PANEL,))],
    control_effect: Annotated[QuantityTensor, DIMENSIONLESS, Domain((TARGET_PANEL,))],
) -> bool:
    """Enforce linearised downwash at all wing and tail collocation points."""
    return (influence * circulation).sum(SOURCE_PANEL) == -speed * (alpha + twist + control_effect * elevator)


@equation
def kutta_joukowski(
    panel_lift: Annotated[QuantityTensor, FORCE, Domain((SOURCE_PANEL,))],
    density: Annotated[Quantity | QuantityTensor, MASS / LENGTH**3],
    speed: Annotated[Quantity | QuantityTensor, LENGTH / TIME],
    circulation: Annotated[QuantityTensor, LENGTH**2 / TIME, Domain((SOURCE_PANEL,))],
    panel_width: Annotated[QuantityTensor, LENGTH, Domain((SOURCE_PANEL,))],
) -> bool:
    """Relate each panel's circulation to its sectional lift."""
    return panel_lift == density * speed * circulation * panel_width


@equation
def integrate_lift(
    lift: Annotated[Quantity | QuantityTensor, FORCE],
    panel_lift: Annotated[QuantityTensor, FORCE, Domain((SOURCE_PANEL,))],
) -> bool:
    """Sum wing and tail lift without mixing the operating-case axis."""
    return lift == panel_lift.sum(SOURCE_PANEL)


@equation
def integrate_pitch_moment(
    pitch_moment: Annotated[Quantity | QuantityTensor, FORCE * LENGTH],
    panel_lift: Annotated[QuantityTensor, FORCE, Domain((SOURCE_PANEL,))],
    moment_arm: Annotated[QuantityTensor, LENGTH, Domain((SOURCE_PANEL,))],
) -> bool:
    """Sum moments about the CG; positive is nose up."""
    return pitch_moment == (moment_arm * panel_lift).sum(SOURCE_PANEL)


@equation
def level_flight(
    lift: Annotated[Quantity | QuantityTensor, FORCE],
    mass: Annotated[Quantity | QuantityTensor, MASS],
    gravity: Annotated[Quantity, LENGTH / TIME**2],
) -> bool:
    """Steady, level flight requires total lift equal to weight."""
    return lift == mass * gravity


MODEL = Implementation(
    WingTailTrim,
    [no_penetration, kutta_joukowski, integrate_lift, integrate_pitch_moment, level_flight],
)
