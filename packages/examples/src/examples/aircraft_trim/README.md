# Acausal wing-and-tail spanwise trim

This example models **four independently solved panel circulations**, two on
the wing and two on the horizontal tail. All four panels interact through
four fixed influence blocks: wing-from-wing, wing-from-tail, tail-from-wing
and tail-from-tail.

It is a *lifting-line-style linear demonstration*, not geometry-derived VLM:
the coefficients are illustrative and have not been calculated using
horseshoe vortices or Biot-Savart.

## Acausal equations

The two wing-target boundary conditions are expressed as a named-tensor
contraction:

    (wing_from_wing * wing_circulation).sum(wing_source)
        + (wing_from_tail * tail_circulation).sum(tail_source)
        == -speed * (alpha + wing_twist)

The tail-target equations are analogous, including the tail control:

    (tail_from_wing * wing_circulation).sum(wing_source)
        + (tail_from_tail * tail_circulation).sum(tail_source)
        == -speed * (alpha + tail_twist + tail_control * elevator)

These equations retain the wing_target and tail_target axes, respectively.

Two scalar balance equations constrain the panel circulations:

    density * speed * (
        (wing_circulation * wing_width).sum(wing_source)
        + (tail_circulation * tail_width).sum(tail_source)
    ) == mass * gravity

    density * speed * (
        (wing_circulation * wing_width * wing_arm).sum(wing_source)
        + (tail_circulation * tail_width * tail_arm).sum(tail_source)
    ) == pitch_moment

The panel forces are independently available from Kutta-Joukowski loading.
For inverse trim the known aircraft weight and zero pitching moment determine
**two wing circulations, two tail circulations, angle of attack and elevator**.
For forward evaluation the two angles determine panel circulations and loads.

Both directions use the same Bilt equations; the tests assemble the independent
six-by-six dense reference in NumPy, never in the model.

## Run

    uv run python -m examples.aircraft_trim.demo
    uv run pytest packages/examples/tests/test_aircraft_trim.py

## Stack

This PR is stacked on the linear tensor operator and mixed scalar/tensor
solver prototype in PR #440. Review that implementation first.

## Limitations

Small-angle steady incompressible flow, prescribed influences rather than
geometry-derived aerodynamics, no induced drag, optimisation or viscous models.
Next: derive influence blocks from vortex geometry, then compare against
AeroSandbox VLM using matching geometry and discretisation.
