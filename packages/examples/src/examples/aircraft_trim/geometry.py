"""Geometry-derived horseshoe-vortex influences for a simple wing and tail.

The aerodynamics are deliberately outside the Bilt equation definitions:
geometry produces known dimensional influence tensors; Bilt determines the
unknown circulation and trim state acausally. No AeroSandbox dependency.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class RectangularSurface:
    """Flat rectangular lifting surface with uniform spanwise panels.

    Positions use x aft, y right, z up, with metres for all lengths.
    Twist values are in radians and are interpolated by absolute span
    fraction, giving a symmetric spanwise twist distribution.
    """

    span_m: float
    chord_m: float
    leading_edge_x_m: float
    height_m: float
    panels: int
    root_twist_rad: float = 0.0
    tip_twist_rad: float = 0.0

    def __post_init__(self) -> None:
        """Reject degenerate geometries before evaluating induced velocities."""
        if not np.isfinite((self.span_m, self.chord_m, self.leading_edge_x_m, self.height_m)).all():
            msg = "Lifting-surface geometry must contain finite coordinates."
            raise ValueError(msg)
        if self.span_m <= 0 or self.chord_m <= 0 or isinstance(self.panels, bool) or self.panels < 1:
            msg = "Lifting-surface span, chord and panel count must be positive."
            raise ValueError(msg)
        if not np.isfinite((self.root_twist_rad, self.tip_twist_rad)).all():
            msg = "Lifting-surface twist must be finite."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class AircraftGeometry:
    """A wing and downstream horizontal tail, with a fixed trailing wake."""

    wing: RectangularSurface = field(
        default_factory=lambda: RectangularSurface(12.0, 1.5, 0.0, 0.0, 2, 0.002, -0.008)
    )
    tail: RectangularSurface = field(
        default_factory=lambda: RectangularSurface(4.5, 0.9, 5.5, 0.5, 2, 0.0, -0.003)
    )
    cg_x_m: float = 1.0
    wake_length_m: float = 400.0

    def __post_init__(self) -> None:
        """Keep the model within the simple downstream-wake assumptions."""
        if not np.isfinite((self.cg_x_m, self.wake_length_m)).all() or self.wake_length_m <= 0:
            msg = "CG coordinate must be finite and wake length positive."
            raise ValueError(msg)
        if self.tail.leading_edge_x_m <= self.wing.leading_edge_x_m:
            msg = "The horizontal tail must be downstream of the wing."
            raise ValueError(msg)


def example_geometry(*, wing_panels: int = 2, tail_panels: int = 2) -> AircraftGeometry:
    """Return the same physical aircraft at different panel resolutions."""
    return AircraftGeometry(
        wing=RectangularSurface(12.0, 1.5, 0.0, 0.0, wing_panels, 0.002, -0.008),
        tail=RectangularSurface(4.5, 0.9, 5.5, 0.5, tail_panels, 0.0, -0.003),
    )


@dataclass(frozen=True, slots=True)
class SurfacePanels:
    """Vortex endpoints, collocation coordinates and load geometry in metres."""

    bound_left: np.ndarray
    bound_right: np.ndarray
    collocation: np.ndarray
    width_m: np.ndarray
    arm_m: np.ndarray
    twist_rad: np.ndarray


def discretise(surface: RectangularSurface, *, cg_x_m: float) -> SurfacePanels:
    """Place bound vortices at quarter-chord and targets at three-quarter chord."""
    edges = np.linspace(-0.5 * surface.span_m, 0.5 * surface.span_m, surface.panels + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bound_x = surface.leading_edge_x_m + 0.25 * surface.chord_m
    target_x = surface.leading_edge_x_m + 0.75 * surface.chord_m
    x_left = np.full(surface.panels, bound_x)
    x_target = np.full(surface.panels, target_x)
    z = np.full(surface.panels, surface.height_m)
    left = np.column_stack((x_left, edges[:-1], z))
    right = np.column_stack((x_left, edges[1:], z))
    collocation = np.column_stack((x_target, centres, z))
    twist = surface.root_twist_rad + (surface.tip_twist_rad - surface.root_twist_rad) * (
        2.0 * np.abs(centres) / surface.span_m
    )
    return SurfacePanels(left, right, collocation, np.diff(edges), cg_x_m - x_left, twist)


def finite_vortex_velocity(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
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
    if squared <= 1e-24 or norm1 <= 1e-12 or norm2 <= 1e-12:
        msg = "Vortex evaluation point lies on or too near a filament."
        raise ValueError(msg)
    return cross * (np.dot(direction, r1 / norm1 - r2 / norm2) / (4.0 * np.pi * squared))


def horseshoe_velocity(
    point: np.ndarray, left: np.ndarray, right: np.ndarray, *, wake_length_m: float
) -> np.ndarray:
    """Velocity due to a horseshoe: left trailing leg, bound span, right leg.

    Positive unit circulation runs from negative to positive span along the
    bound vortex. Trailing legs extend in +x for a stationary straight wake.
    """
    wake = np.array([wake_length_m, 0.0, 0.0])
    return (
        finite_vortex_velocity(point, left + wake, left)
        + finite_vortex_velocity(point, left, right)
        + finite_vortex_velocity(point, right, right + wake)
    )


def influence_block(
    targets: SurfacePanels, sources: SurfacePanels, *, wake_length_m: float
) -> np.ndarray:
    """Vertical induced velocity per unit circulation, target-by-source (m^-1)."""
    return np.array(
        [
            [
                horseshoe_velocity(target, left, right, wake_length_m=wake_length_m)[2]
                for left, right in zip(sources.bound_left, sources.bound_right, strict=True)
            ]
            for target in targets.collocation
        ]
    )


@dataclass(frozen=True, slots=True)
class InfluenceGeometry:
    """Four physically coupled source-to-target influence blocks, in m^-1."""

    wing_from_wing: np.ndarray
    wing_from_tail: np.ndarray
    tail_from_wing: np.ndarray
    tail_from_tail: np.ndarray
    wing: SurfacePanels
    tail: SurfacePanels


def calculate_influences(geometry: AircraftGeometry) -> InfluenceGeometry:
    """Build wing/tail influence blocks from horseshoe-vortex geometry."""
    wing = discretise(geometry.wing, cg_x_m=geometry.cg_x_m)
    tail = discretise(geometry.tail, cg_x_m=geometry.cg_x_m)
    wake = geometry.wake_length_m
    return InfluenceGeometry(
        wing_from_wing=influence_block(wing, wing, wake_length_m=wake),
        wing_from_tail=influence_block(wing, tail, wake_length_m=wake),
        tail_from_wing=influence_block(tail, wing, wake_length_m=wake),
        tail_from_tail=influence_block(tail, tail, wake_length_m=wake),
        wing=wing,
        tail=tail,
    )
