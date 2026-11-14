"""Small prescribed influence matrix for the wing-and-tail example."""

from bilt import Quantity, QuantityTensor, tensor
from bilt.units import kilogram, meter, radian, second
from bilt.units.units import newton

from .model import SOURCE_PANEL, TARGET_PANEL, WingTailTrim

# Same four physical panels on both axes: wing left/right then tail left/right.
PANEL_LABELS = ("wing_left", "wing_right", "tail_left", "tail_right")


def example_aircraft(
    *,
    mass: Quantity | QuantityTensor | None = None,
    alpha: Quantity | QuantityTensor | None = None,
    elevator: Quantity | QuantityTensor | None = None,
    pitch_moment: Quantity | QuantityTensor | None = None,
) -> WingTailTrim:
    """Construct a demo aircraft with a prescribed, invertible influence matrix.

    This illustrative matrix is not calculated from Biot-Savart or VLM.
    Each source/target panel shares the same index ordering, although the two
    named axes are deliberately distinct to express the matrix contraction.
    """
    known: dict[str, object] = {
        "influence": tensor(
            [
                [-0.35, -0.07, -0.01, -0.01],
                [-0.07, -0.35, -0.01, -0.01],
                [-0.06, -0.06, -0.60, -0.07],
                [-0.06, -0.06, -0.07, -0.60],
            ],
            over=(TARGET_PANEL, SOURCE_PANEL),
            unit=1 / meter,
        ),
        "panel_width": tensor([4.0, 4.0, 2.0, 2.0], over=SOURCE_PANEL, unit=meter),
        "moment_arm": tensor([0.5, 0.5, -4.0, -4.0], over=SOURCE_PANEL, unit=meter),
        "twist": tensor([0.0] * 4, over=TARGET_PANEL, unit=radian),
        "control_effect": tensor([0.0, 0.0, 1.0, 1.0], over=TARGET_PANEL, unit=radian / radian),
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
    """Set the mass and zero moment, leaving alpha and elevator unknown."""
    return example_aircraft(mass=mass_kg * kilogram, pitch_moment=0.0 * newton * meter)
