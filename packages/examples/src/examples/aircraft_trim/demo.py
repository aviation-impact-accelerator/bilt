"""Run the small acausal wing-and-tail trim example."""

# ruff: noqa: T201  # This script deliberately prints the demonstration results.

from bilt.units import degree

from .fixtures import example_aircraft, trim_case
from .model import MODEL


def main() -> None:
    """Evaluate trim and a forward calculation with the same equations."""
    with MODEL:
        trimmed = trim_case()
        alpha = trimmed.alpha
        elevator = trimmed.elevator
        circulation = trimmed.circulation
        print(f"Trim angle of attack: {alpha.convert_to(degree).value:.3f} deg")
        print(f"Trim tail deflection: {elevator.convert_to(degree).value:.3f} deg")
        print(f"Circulation (m²/s): {circulation.values}")

        forward = example_aircraft(alpha=alpha, elevator=elevator)
        print(f"Forward lift (N): {forward.lift.value:.1f}")
        print(f"Forward pitching moment (N m): {forward.pitch_moment.value:.3e}")


if __name__ == "__main__":
    main()
