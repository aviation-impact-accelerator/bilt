"""Geometry and numerical reference tests for acausal wing-tail trim."""

import numpy as np
import pytest
from examples.aircraft_trim.aircraft import MODEL
from examples.aircraft_trim.fixtures import example_aircraft, forward_case, trim_case
from examples.aircraft_trim.geometry import (
    CARTESIAN,
    GEOMETRY_MODEL,
    PANEL,
    PANEL_MODEL,
    SEGMENT,
    SURFACE_MODEL,
    VORTEX_MODEL,
    AircraftGeometry,
    PanelResolution,
    RectangularSurface,
    SurfacePanels,
    VortexInfluence,
    example_geometry,
)
from examples.aircraft_trim.model import TAIL_SOURCE, TAIL_TARGET, WING_SOURCE, WING_TARGET

from bilt import Quantity, QuantityTensor, tensor
from bilt.units import degree, fraction, kilogram, meter, newton, radian, second

_REFERENCE_LENGTH_TOLERANCE = 1e-12
_REFERENCE_CROSS_TOLERANCE = _REFERENCE_LENGTH_TOLERANCE**4


def _discretise(surface: RectangularSurface, *, cg_x: Quantity, panels: int) -> SurfacePanels:
    """Generate the numerical panel axis and evaluate dimensional geometry."""
    with SURFACE_MODEL:
        return SurfacePanels(
            span=surface.span,
            bound_x=surface.bound_x,
            target_x=surface.target_x,
            height=surface.height,
            cg_x=cg_x,
            root_twist=surface.root_twist,
            tip_twist=surface.tip_twist,
            panel_index=tensor(np.arange(panels), over=PANEL, unit=fraction),
            panel_count=float(panels) * fraction,
        )


def _evaluate_vortices(point: QuantityTensor, start: QuantityTensor, end: QuantityTensor) -> QuantityTensor:
    """Exercise the standalone vortex implementation through its public fields."""
    with VORTEX_MODEL:
        return VortexInfluence(point=point, start=start, end=end).influence


def _influences(geometry: AircraftGeometry, *, resolution: PanelResolution) -> dict[str, QuantityTensor]:
    """Read influence outputs and align the independent reference array order."""
    with MODEL:
        aircraft = example_aircraft(geometry=geometry, resolution=resolution)
        blocks = {
            "wing_from_wing": (aircraft.wing_from_wing, WING_TARGET, WING_SOURCE),
            "wing_from_tail": (aircraft.wing_from_tail, WING_TARGET, TAIL_SOURCE),
            "tail_from_wing": (aircraft.tail_from_wing, TAIL_TARGET, WING_SOURCE),
            "tail_from_tail": (aircraft.tail_from_tail, TAIL_TARGET, TAIL_SOURCE),
        }
    return {
        name: tensor(
            block.convert_to(meter**-1).values.transpose(tuple(block.domain.dims.index(axis) for axis in (target, source))),
            over=(target, source),
            unit=meter**-1,
        )
        for name, (block, target, source) in blocks.items()
    }


def _reference_segment_velocity(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Biot-Savart velocity in m^-1 for a straight unit-circulation filament.

    Formula uses the finite-segment cross product and endpoint factors.
    Near-singular configurations are rejected rather than regularised
    silently; all generated collocation points avoid the vortex lines.
    """
    r1 = point - start
    r2 = point - end
    direction = end - start
    cross = np.cross(r1, r2)
    squared = float(np.dot(cross, cross))
    norm1 = float(np.linalg.norm(r1))
    norm2 = float(np.linalg.norm(r2))
    if (
        squared <= _REFERENCE_CROSS_TOLERANCE
        or norm1 <= _REFERENCE_LENGTH_TOLERANCE
        or norm2 <= _REFERENCE_LENGTH_TOLERANCE
    ):
        msg = "Vortex evaluation point lies on or too near a filament."
        raise ValueError(msg)
    return cross * (np.dot(direction, r1 / norm1 - r2 / norm2) / (4.0 * np.pi * squared))


def _reference_horseshoe_velocity(
    point: np.ndarray, left: np.ndarray, right: np.ndarray, *, wake_length_m: float
) -> np.ndarray:
    """Velocity due to a horseshoe: left trailing leg, bound span, right leg.

    Positive unit circulation runs from negative to positive span along the
    bound vortex. Trailing legs extend in +x for a stationary straight wake.
    """
    wake = np.array([wake_length_m, 0.0, 0.0])
    return (
        _reference_segment_velocity(point, left + wake, left)
        + _reference_segment_velocity(point, left, right)
        + _reference_segment_velocity(point, right, right + wake)
    )


def _reference_surface(  # noqa: PLR0913, PLR0917 - independent physical inputs.
    span: float, chord: float, leading_edge: float, height: float, count: int, root: float, tip: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct the reference geometry directly in NumPy."""
    edges = np.linspace(-0.5 * span, 0.5 * span, count + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bound_x = leading_edge + 0.25 * chord
    target_x = leading_edge + 0.75 * chord
    left = np.column_stack((np.full(count, bound_x), edges[:-1], np.full(count, height)))
    right = np.column_stack((np.full(count, bound_x), edges[1:], np.full(count, height)))
    points = np.column_stack((np.full(count, target_x), centres, np.full(count, height)))
    return (
        left,
        right,
        points,
        np.diff(edges),
        np.full(count, 1.0 - bound_x),
        root + (tip - root) * 2.0 * abs(centres) / span,
    )


def _reference_operator(
    wing_panels: int, tail_panels: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble a dense system from independent geometry and vortex calculations."""
    wing = _reference_surface(12.0, 1.5, 0.0, 0.0, wing_panels, 0.002, -0.008)
    tail = _reference_surface(4.5, 0.9, 5.5, 0.5, tail_panels, 0.0, -0.003)
    left = np.vstack((wing[0], tail[0]))
    right = np.vstack((wing[1], tail[1]))
    points = np.vstack((wing[2], tail[2]))
    a = np.array(
        [
            [
                _reference_horseshoe_velocity(point, first, last, wake_length_m=400.0)[2]
                for first, last in zip(left, right, strict=True)
            ]
            for point in points
        ]
    )
    widths = np.r_[wing[3], tail[3]]
    arms = np.r_[wing[4], tail[4]]
    twist = np.r_[wing[5], tail[5]]
    control = np.r_[np.zeros(wing_panels), np.ones(tail_panels)]
    return a, widths, arms, twist, control


def _reference_trim(mass: float, wing_panels: int, tail_panels: int) -> np.ndarray:
    """Direct NumPy solve: N aerodynamic rows and two global trim constraints."""
    a, widths, arms, twist, control = _reference_operator(wing_panels, tail_panels)
    n = wing_panels + tail_panels
    matrix = np.zeros((n + 2, n + 2))
    matrix[:n, :n] = a
    matrix[:n, n] = 70.0
    matrix[:n, n + 1] = 70.0 * control
    matrix[n, :n] = 1.225 * 70.0 * widths
    matrix[n + 1, :n] = 1.225 * 70.0 * widths * arms
    return np.linalg.solve(matrix, np.r_[-70.0 * twist, mass * 9.81, 0.0])


def _reference_forward(
    alpha: float, elevator: float, wing_panels: int, tail_panels: int
) -> tuple[np.ndarray, float, float]:
    """Direct NumPy solve for circulation and integrated lift/moment."""
    a, widths, arms, twist, control = _reference_operator(wing_panels, tail_panels)
    gamma = np.linalg.solve(a, -70.0 * (alpha + twist + control * elevator))
    loads = 1.225 * 70.0 * gamma * widths
    return gamma, float(loads.sum()), float(np.dot(loads, arms))


def _combine(wing: QuantityTensor, tail: QuantityTensor) -> np.ndarray:
    return np.r_[wing.convert_to(meter**2 / second).values, tail.convert_to(meter**2 / second).values]


@pytest.mark.parametrize(("wing_panels", "tail_panels"), ((2, 2), (6, 4), (12, 6), (5, 3)))
def test_geometry_based_forward_matches_numpy(wing_panels: int, tail_panels: int) -> None:
    """Bilt solves each source-panel circulation from physical vortex influence."""
    geometry = example_geometry()
    resolution = PanelResolution(wing=wing_panels, tail=tail_panels)
    with MODEL:
        aircraft = forward_case(3.0, -1.0, geometry=geometry, resolution=resolution)
        wing = aircraft.wing_circulation
        tail = aircraft.tail_circulation
        lift = aircraft.lift
        moment = aircraft.pitch_moment

    expected, expected_lift, expected_moment = _reference_forward(
        np.deg2rad(3.0), np.deg2rad(-1.0), wing_panels, tail_panels
    )
    assert isinstance(wing, QuantityTensor)
    assert isinstance(tail, QuantityTensor)
    assert wing.domain.dims == (WING_SOURCE,)
    assert tail.domain.dims == (TAIL_SOURCE,)
    np.testing.assert_allclose(_combine(wing, tail), expected, rtol=1e-6, atol=1e-7)
    assert np.ptp(expected) > 0.0
    assert lift.convert_to(newton).value == pytest.approx(expected_lift, rel=1e-6)
    assert moment.convert_to(newton * meter).value == pytest.approx(expected_moment, rel=1e-6)


@pytest.mark.parametrize(("wing_panels", "tail_panels"), ((2, 2), (6, 4), (12, 6), (5, 3)))
def test_geometry_based_inverse_trim_matches_numpy(wing_panels: int, tail_panels: int) -> None:
    """N panel unknowns plus two global unknowns form an (N+2)-square problem."""
    geometry = example_geometry()
    resolution = PanelResolution(wing=wing_panels, tail=tail_panels)
    with MODEL:
        aircraft = trim_case(550.0, geometry=geometry, resolution=resolution)
        wing = aircraft.wing_circulation
        tail = aircraft.tail_circulation
        alpha = aircraft.alpha
        elevator = aircraft.elevator

    expected = _reference_trim(550.0, wing_panels, tail_panels)
    assert isinstance(wing, QuantityTensor)
    assert isinstance(tail, QuantityTensor)
    np.testing.assert_allclose(_combine(wing, tail), expected[:-2], rtol=1e-6, atol=1e-7)
    assert alpha.convert_to(degree).value == pytest.approx(np.rad2deg(expected[-2]), rel=1e-6)
    assert elevator.convert_to(degree).value == pytest.approx(np.rad2deg(expected[-1]), rel=1e-6)


def test_trim_angles_reproduce_lift_and_zero_pitch() -> None:
    """Re-evaluate the same equations forwards using inverse-trim angles."""
    geometry = example_geometry()
    resolution = PanelResolution(wing=6, tail=4)
    with MODEL:
        trimmed = trim_case(550.0, geometry=geometry, resolution=resolution)
        forward = example_aircraft(
            alpha=trimmed.alpha, elevator=trimmed.elevator, geometry=geometry, resolution=resolution
        )
        lift = forward.lift
        moment = forward.pitch_moment
        wing_load = forward.wing_panel_lift
        tail_load = forward.tail_panel_lift
        mass = forward.mass

    assert lift.convert_to(newton).value == pytest.approx(550.0 * 9.81, rel=1e-6)
    assert moment.convert_to(newton * meter).value == pytest.approx(0.0, abs=1e-6)
    assert mass.convert_to(kilogram).value == pytest.approx(550.0, rel=1e-6)
    assert isinstance(wing_load, QuantityTensor)
    assert isinstance(tail_load, QuantityTensor)
    assert wing_load.domain.dims == (WING_SOURCE,)
    assert tail_load.domain.dims == (TAIL_SOURCE,)
    gamma = _reference_trim(550.0, 6, 4)[:-2]
    _, widths, arms, _, _ = _reference_operator(6, 4)
    expected_loads = 1.225 * 70.0 * gamma * widths
    loads = np.r_[wing_load.convert_to(newton).values, tail_load.convert_to(newton).values]
    np.testing.assert_allclose(loads, expected_loads, rtol=1e-6, atol=1e-6)
    assert loads.sum() == pytest.approx(lift.convert_to(newton).value, rel=1e-6)
    assert np.dot(loads, arms) == pytest.approx(moment.convert_to(newton * meter).value, abs=1e-6)


def test_prescribed_lift_instead_of_mass() -> None:
    """The same equations can trim at a specified lift and recover mass."""
    with MODEL:
        aircraft = example_aircraft(lift=550.0 * 9.81 * newton, pitch_moment=0.0 * newton * meter)
        alpha = aircraft.alpha
        elevator = aircraft.elevator
        mass = aircraft.mass

    expected = _reference_trim(550.0, 2, 2)
    assert alpha.convert_to(degree).value == pytest.approx(np.rad2deg(expected[-2]), rel=1e-6)
    assert elevator.convert_to(degree).value == pytest.approx(np.rad2deg(expected[-1]), rel=1e-6)
    assert mass.convert_to(kilogram).value == pytest.approx(550.0, rel=1e-6)


@pytest.mark.parametrize("mass", (420.0, 550.0, 680.0))
def test_inverse_trim_for_multiple_weights(mass: float) -> None:
    """Check the solver with different loads at a fixed geometry."""
    with MODEL:
        aircraft = trim_case(mass)
        wing = aircraft.wing_circulation
        tail = aircraft.tail_circulation
    np.testing.assert_allclose(_combine(wing, tail), _reference_trim(mass, 2, 2)[:-2], rtol=1e-6)


def test_finite_segment_biot_savart_against_analytic_value() -> None:
    """A centred unit straight vortex has a known velocity at unit distance."""
    influence = _evaluate_vortices(
        tensor([0.0, 1.0, 0.0], over=CARTESIAN, unit=meter),
        tensor([[-1.0, 0.0, 0.0]], over=(SEGMENT, CARTESIAN), unit=meter),
        tensor([[1.0, 0.0, 0.0]], over=(SEGMENT, CARTESIAN), unit=meter),
    )
    np.testing.assert_allclose(influence.convert_to(meter**-1).values, 1.0 / (2.0 * np.pi * np.sqrt(2.0)), atol=1e-12)


def test_singular_vortex_is_rejected() -> None:
    point = tensor([0.0, 0.0, 0.0], over=CARTESIAN, unit=meter)
    start = tensor([[-1.0, 0.0, 0.0]], over=(SEGMENT, CARTESIAN), unit=meter)
    end = tensor([[1.0, 0.0, 0.0]], over=(SEGMENT, CARTESIAN), unit=meter)
    with pytest.raises(ValueError, match="too near a filament"):
        _evaluate_vortices(point, start, end)


@pytest.mark.parametrize(("wing_panels", "tail_panels"), ((2, 2), (6, 4), (12, 6), (5, 3)))
def test_influence_blocks_match_numpy(wing_panels: int, tail_panels: int) -> None:
    data = _influences(example_geometry(), resolution=PanelResolution(wing=wing_panels, tail=tail_panels))
    expected, _, _, _, _ = _reference_operator(wing_panels, tail_panels)
    for actual, reference in (
        (data["wing_from_wing"], expected[:wing_panels, :wing_panels]),
        (data["wing_from_tail"], expected[:wing_panels, wing_panels:]),
        (data["tail_from_wing"], expected[wing_panels:, :wing_panels]),
        (data["tail_from_tail"], expected[wing_panels:, wing_panels:]),
    ):
        np.testing.assert_allclose(actual.convert_to(meter**-1).values, reference, rtol=1e-9, atol=1e-12)


def test_symmetric_geometry_gives_symmetric_spanwise_influence() -> None:
    """Mirror symmetry provides a geometry-based influence regression."""
    data = _influences(example_geometry(), resolution=PanelResolution(wing=6, tail=4))
    ww = data["wing_from_wing"].convert_to(meter**-1).values
    tt = data["tail_from_tail"].convert_to(meter**-1).values
    wt = data["wing_from_tail"].convert_to(meter**-1).values
    np.testing.assert_allclose(ww, ww[::-1, ::-1], atol=1e-12)
    np.testing.assert_allclose(tt, tt[::-1, ::-1], atol=1e-12)
    np.testing.assert_allclose(wt, wt[::-1, ::-1], atol=1e-12)
    assert np.all(np.diag(ww) < 0.0)


def test_wake_length_convergence_and_cross_surface_coupling() -> None:
    """Farther wake truncation should not dramatically alter a near-field matrix."""
    near = example_geometry()
    far = example_geometry(wake_length=800.0 * meter)
    resolution = PanelResolution(wing=6, tail=4)
    first = _influences(near, resolution=resolution)
    second = _influences(far, resolution=resolution)
    np.testing.assert_allclose(first["wing_from_wing"].values, second["wing_from_wing"].values, rtol=2e-4, atol=2e-6)
    assert np.any(abs(first["tail_from_wing"].values) > 0.0)


@pytest.mark.parametrize(
    ("span", "chord", "panels"),
    ((0.0, 1.0, 2), (2.0, 0.0, 2), (2.0, 1.0, 0)),
)
def test_invalid_surface_geometry_is_rejected(span: float, chord: float, panels: int) -> None:
    """Reject invalid geometry early, rather than entering numerical assembly."""
    surface = RectangularSurface(
        span=span * meter,
        chord=chord * meter,
        leading_edge_x=0.0 * meter,
        height=0.0 * meter,
        root_twist=0.0 * radian,
        tip_twist=0.0 * radian,
    )
    geometry = AircraftGeometry(
        wing_span=surface.span,
        wing_chord=surface.chord,
        wing_leading_edge_x=surface.leading_edge_x,
        wing_height=surface.height,
        wing_root_twist=surface.root_twist,
        wing_tip_twist=surface.tip_twist,
        tail_span=4.5 * meter,
        tail_chord=0.9 * meter,
        tail_leading_edge_x=5.5 * meter,
        tail_height=0.5 * meter,
        tail_root_twist=0.0 * radian,
        tail_tip_twist=0.0 * radian,
        cg_x=1.0 * meter,
        wake_length=400.0 * meter,
    )
    with MODEL, pytest.raises(ValueError, match="span, chord and panel count|Panel count"):
        _ = example_aircraft(geometry=geometry, resolution=PanelResolution(wing=panels, tail=2)).wing_width


def test_surface_geometry_is_acausal() -> None:
    with SURFACE_MODEL:
        surface = RectangularSurface(span=12.0 * meter, area=18.0 * meter**2)
        assert surface.chord.convert_to(meter).value == pytest.approx(1.5)
        assert surface.aspect_ratio.convert_to(fraction).value == pytest.approx(8.0)


def test_aircraft_geometry_reuses_surface_model() -> None:
    with GEOMETRY_MODEL:
        geometry = example_geometry()
        assert geometry.wing_area.convert_to(meter**2).value == pytest.approx(18.0)
        assert geometry.tail_area.convert_to(meter**2).value == pytest.approx(4.05)


@pytest.mark.parametrize("count", (2, 5, 6, 12))
def test_panel_geometry_matches_numpy(count: int) -> None:
    """Independently check the migrated geometry equations at several resolutions."""
    surface = RectangularSurface(
        span=12.0 * meter,
        chord=1.5 * meter,
        leading_edge_x=0.0 * meter,
        height=0.0 * meter,
        root_twist=0.002 * radian,
        tip_twist=-0.008 * radian,
    )
    panels = _discretise(surface, cg_x=1.0 * meter, panels=count)
    edges = np.linspace(-6.0, 6.0, count + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    with PANEL_MODEL:
        np.testing.assert_allclose(panels.left_y.convert_to(meter).values, edges[:-1], atol=1e-12)
        np.testing.assert_allclose(panels.right_y.convert_to(meter).values, edges[1:], atol=1e-12)
        np.testing.assert_allclose(panels.centre_y.convert_to(meter).values, centres, atol=1e-12)
        np.testing.assert_allclose(panels.width.convert_to(meter).values, np.diff(edges), atol=1e-12)
        np.testing.assert_allclose(panels.twist.convert_to(radian).values, 0.002 - 0.01 * abs(centres) / 6.0)
        assert panels.arm.convert_to(meter).value == pytest.approx(0.625)
        assert panels.bound_x.convert_to(meter).value == pytest.approx(0.375)
        assert panels.target_x.convert_to(meter).value == pytest.approx(1.125)


def test_composed_aircraft_exposes_geometry_without_trim_inputs() -> None:
    """Geometry outputs are independently requestable from the same aircraft."""
    with MODEL:
        aircraft = example_aircraft(resolution=PanelResolution(wing=5, tail=3))
        assert aircraft.wing_area.convert_to(meter**2).value == pytest.approx(18.0)
        assert aircraft.tail_area.convert_to(meter**2).value == pytest.approx(4.05)
        assert set(aircraft.wing_from_tail.domain.dims) == {WING_TARGET, TAIL_SOURCE}
        assert aircraft.wing_width.domain.dims == (WING_SOURCE,)
        assert aircraft.tail_twist.domain.dims == (TAIL_TARGET,)


def test_composed_aircraft_validates_wake_on_influence_path() -> None:
    with MODEL, pytest.raises(ValueError, match="wake length positive"):
        _ = example_aircraft(geometry=example_geometry(wake_length=0.0 * meter)).wing_from_wing
