# Geometry-based acausal wing-and-tail lifting-line example

This example solves individual bound-vortex circulations on a finite wing and
horizontal tail. It is a **horseshoe-vortex lifting-line-style model**, with
a geometric influence matrix rather than prescribed aerodynamic coefficients.

The wing and tail have separate named source and target dimensions. Every
source vortex induces normal velocity at every target collocation point,
including aerodynamic interference between the wing and tail.

## Geometry and assumptions

Each lifting surface is a flat rectangular wing discretised spanwise. Bound
vortices sit at quarter-chord, collocation points at three-quarter-chord,
and the two trailing wake legs run downstream (+x) over a finite length.
The model uses x aft, y right and z up, with positive circulation directed
from negative to positive span on the bound segment. Induced velocities are
calculated with the finite-segment Biot-Savart formula, in units of 1/metre
per unit circulation.

A normal-velocity approximation is used for small incidence angles.
Panel lift uses the sectional Kutta-Joukowski relationship. The result is
not a full vortex-lattice or viscous aerodynamic model: chordwise panels,
wake roll-up, induced drag, compressibility, stall and fuselage effects are
outside scope. Geometric influences are generated once as model inputs,
rather than solved inside an equation.

## Acausal Bilt equations

The four influence tensors have axes:
wing-from-wing (wing_target, wing_source), wing-from-tail
(wing_target, tail_source), tail-from-wing (tail_target, wing_source) and
tail-from-tail (tail_target, tail_source).

The wing boundary condition is:

    (wing_from_wing * wing_circulation).sum(wing_source)
        + (wing_from_tail * tail_circulation).sum(tail_source)
        == -speed * (alpha + wing_twist)

The tail boundary condition also includes elevator and tail twist. Global
longitudinal trim imposes:

    total panel lift == mass * gravity
    sum(panel lift * moment arm) == pitch_moment

In inverse trim, Bilt determines all wing and tail circulations, the
aircraft angle of attack and the elevator deflection from a prescribed
weight and zero pitching moment. Forward evaluation instead specifies both
angles and determines the panel circulations and forces.

With N wing and M tail panels, inverse trim is an (N+M+2)-unknown
linear problem. A separate NumPy assembly is used only as an independent
test reference, not in the acausal model.

## Run

From the Bilt repository root:

    uv run python -m examples.aircraft_trim.demo
    uv run pytest packages/examples/tests/test_aircraft_trim.py

The demo uses 6 wing and 4 tail panels, with a 12-by-12 inverse trim system. Tests cover the original 2+2
configuration, 6+4, and several geometric invariants. The panel counts are
parameters in the geometry fixture, not hard-coded in the Bilt equations.

## Dependency and future work

This is stacked on the general source-to-target/mixed-domain linear solver
prototype in PR #440, and should remain a draft example until the solver's
structural matching heuristic has been properly generalised.

Possible follow-ups: refine spanwise discretisation and convergence, introduce
chordwise vortex panels, model variable chord and sweep, and compare against a
matching AeroSandbox configuration with the same horseshoe and wake conventions.
