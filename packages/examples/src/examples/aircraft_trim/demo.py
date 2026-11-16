"""Run the four-panel acausal wing-and-tail longitudinal trim example."""

# ruff: noqa: T201  # Executable example deliberately prints results.

from bilt.units import degree, meter, second

from .fixtures import forward_case, trim_case
from .model import MODEL


def main() -> None:
    """Solve four panel circulations and two angles, then verify forwards."""
    with MODEL:
        trimmed = trim_case()
        alpha = trimmed.alpha
        elevator = trimmed.elevator
        print(f"Wing circulation (m²/s): {trimmed.wing_circulation.convert_to(meter**2 / second).values}")
        print(f"Tail circulation (m²/s): {trimmed.tail_circulation.convert_to(meter**2 / second).values}")
        print(f"Trim angle of attack: {alpha.convert_to(degree).value:.3f} degrees")
        print(f"Tail deflection: {elevator.convert_to(degree).value:.3f} degrees")

        forward = forward_case(alpha.convert_to(degree).value, elevator.convert_to(degree).value)
        print(f"Wing panel lift (N): {forward.wing_panel_lift.values}")
        print(f"Tail panel lift (N): {forward.tail_panel_lift.values}")
        print(f"Total lift (N): {forward.lift.value:.2f}")
        print(f"Pitch moment (N m): {forward.pitch_moment.value:.4e}")


if __name__ == "__main__":
    main()
