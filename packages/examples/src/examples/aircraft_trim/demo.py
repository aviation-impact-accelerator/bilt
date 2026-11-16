"""Run a geometry-based wing-and-tail acausal circulation/trim example."""

# ruff: noqa: T201  # Executable example intentionally prints results.

from bilt.units import degree, meter, second

from .fixtures import forward_case, trim_case
from .geometry import example_geometry
from .model import MODEL


def main() -> None:
    """Solve the larger ten-panel discretisation and verify forwards."""
    geometry = example_geometry(wing_panels=6, tail_panels=4)
    with MODEL:
        trimmed = trim_case(geometry=geometry)
        alpha = trimmed.alpha
        elevator = trimmed.elevator
        print(f"Wing circulation (m²/s): {trimmed.wing_circulation.convert_to(meter**2 / second).values}")
        print(f"Tail circulation (m²/s): {trimmed.tail_circulation.convert_to(meter**2 / second).values}")
        print(f"Angle of attack: {alpha.convert_to(degree).value:.3f} degrees")
        print(f"Tail deflection: {elevator.convert_to(degree).value:.3f} degrees")
        forward = forward_case(
            alpha.convert_to(degree).value,
            elevator.convert_to(degree).value,
            geometry=geometry,
        )
        print(f"Wing panel lift (N): {forward.wing_panel_lift.values}")
        print(f"Tail panel lift (N): {forward.tail_panel_lift.values}")
        print(f"Total lift (N): {forward.lift.value:.2f}")
        print(f"Pitch moment (N m): {forward.pitch_moment.value:.4e}")


if __name__ == "__main__":
    main()
