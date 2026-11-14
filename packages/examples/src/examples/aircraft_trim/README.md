# Acausal wing-and-tail longitudinal trim

This example exercises Bilt's acausal equations, unit-aware named tensors and simultaneous linear solves. It is deliberately **not** a vortex-lattice method (VLM) or an optimisation example.

The prescribed four-panel influence matrix covers two wing panels and two horizontal-tail panels. An influence coefficient maps the circulation at a source panel to normal velocity at a target panel, including wing-tail interference. Both axes refer to panels in the same order but have separate names:

    influence           [target_panel, source_panel]   1/m
    circulation         [case?, source_panel]          m²/s
    (influence * circulation).sum(source_panel)
                        [case?, target_panel]          m/s

For small angles in radians, fixed coefficients, steady incompressible flow and a rigid aircraft:

    sum_j A[i,j] * Gamma[j] = -V * (alpha + twist[i] + control_effect[i] * elevator)
    panel_lift[j] = density * V * Gamma[j] * panel_width[j]
    lift = sum_j panel_lift[j] = mass * gravity
    pitch_moment = sum_j moment_arm[j] * panel_lift[j]

Positive pitching moment is nose-up; wing panels are ahead of the centre of gravity and tail panels behind it. The control effect is zero on the wing and unity on the tail. This toy influence matrix is specified numerically, **not derived from aircraft geometry**. It is not intended to predict real trim angles, induced drag, viscous effects or aerodynamic stability.

## Run

From the repository root:

    uv run python -m examples.aircraft_trim.demo
    uv run pytest packages/examples/tests/test_aircraft_trim.py

The inverse question supplies aircraft mass and zero pitching moment, then asks for circulation, angle of attack and tail deflection. The forward question supplies those two angles and asks for aerodynamic lift and pitching moment, without changing the equations.

The tests compare both directions against an independently assembled dense NumPy system and exercise independent cases using a named `case` dimension. If the cross-axis contraction is not yet supported by Bilt's structural solver, the test failures will identify exactly which general-purpose tensor-linear-solve capability is missing.

## Intended follow-up

Implement a directional geometry step to generate influence coefficients from horseshoe vortices, then compare a like-for-like discretisation against AeroSandbox VLM. Keep this geometry calculation separate from the acausal aerodynamic and trim relationships.
