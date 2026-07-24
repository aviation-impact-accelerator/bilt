"""Scrambled Sobol sampling broadcast over a Bilt mission model.

The only explicit loop constructs independent Sobol randomizations. The model
itself is written as scalar transforms and lifted by Bilt over the union of the
``scramble``, ``sample``, ``route``, and ``payload_case`` domains.

Boolean comparisons are performed after Bilt evaluation because the current
BiltTensor implementation supports arithmetic lifting but does not yet support
comparison operators such as ``<=``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, lognorm, norm, qmc

from bilt._backend._python import PythonBackend
from bilt.core._system._decorator import system
from bilt.core._system._field import field
from bilt.core._system._implementation import Implementation
from bilt.core._system._interface import Interface
from bilt.core._tensor._tensor import BiltTensor, tensor
from bilt.core._transform import transform
from bilt.symbolic._dim import Dim


@dataclass(frozen=True)
class QMCInputs:
    """Physical inputs generated from independently scrambled Sobol designs."""

    specific_energy_kwh_per_t: np.ndarray
    propulsive_efficiency: np.ndarray
    energy_multiplier: np.ndarray


def scrambled_sobol_inputs(
    *,
    scrambles: int = 8,
    power: int = 10,
    seed: int = 1_000,
) -> QMCInputs:
    """Generate independent scrambled Sobol designs of size ``2**power``."""

    samples = 2**power
    uniforms = np.empty((scrambles, samples, 3), dtype=float)

    # This is the only explicit evaluation-related loop. Each iteration creates
    # one independently randomized Sobol design.
    for scramble_index in range(scrambles):
        engine = qmc.Sobol(
            d=3,
            scramble=True,
            seed=seed + scramble_index,
        )
        uniforms[scramble_index] = engine.random_base2(power)

    # Avoid passing exactly zero or one into inverse distribution functions.
    eps = np.finfo(float).eps
    uniforms = np.clip(uniforms, eps, 1.0 - eps)

    return QMCInputs(
        specific_energy_kwh_per_t=norm.ppf(
            uniforms[..., 0],
            loc=450.0,
            scale=45.0,
        ),
        propulsive_efficiency=(
            0.70
            + 0.25
            * beta.ppf(
                uniforms[..., 1],
                a=12.0,
                b=3.0,
            )
        ),
        energy_multiplier=lognorm.ppf(
            uniforms[..., 2],
            s=0.08,
            scale=1.0,
        ),
    )


@system
class Mission(metaclass=Interface):
    """Battery-electric mission model evaluated over named Bilt domains."""

    # QMC inputs over (scramble, sample).
    specific_energy_kwh_per_t: BiltTensor = field()
    propulsive_efficiency: BiltTensor = field()
    energy_multiplier: BiltTensor = field()

    # Deterministic design inputs over route and payload_case.
    distance_km: BiltTensor = field()
    payload_t: BiltTensor = field()

    # Scalar configuration inputs.
    empty_mass_t: float = field()
    energy_per_km_t: float = field()
    usable_fraction: float = field()
    max_battery_mass_t: float = field()

    # Broadcast derived values.
    gross_mass_t: BiltTensor = field()
    shaft_energy_kwh: BiltTensor = field()
    electrical_energy_kwh: BiltTensor = field()
    battery_mass_t: BiltTensor = field()
    battery_margin_t: BiltTensor = field()


@transform
def gross_mass_t(
    payload_t: float,
    empty_mass_t: float,
) -> float:
    """Calculate aircraft mass for one payload case."""

    return empty_mass_t + payload_t


@transform
def shaft_energy_kwh(
    energy_multiplier: float,
    distance_km: float,
    gross_mass_t: float,
    energy_per_km_t: float,
) -> float:
    """Calculate required shaft energy for one scalar scenario."""

    return (
        energy_multiplier
        * distance_km
        * gross_mass_t
        * energy_per_km_t
    )


@transform
def electrical_energy_kwh(
    shaft_energy_kwh: float,
    propulsive_efficiency: float,
) -> float:
    """Convert shaft-energy demand into electrical-energy demand."""

    return shaft_energy_kwh / propulsive_efficiency


@transform
def battery_mass_t(
    electrical_energy_kwh: float,
    specific_energy_kwh_per_t: float,
    usable_fraction: float,
) -> float:
    """Calculate the required battery mass."""

    return electrical_energy_kwh / (
        specific_energy_kwh_per_t * usable_fraction
    )


@transform
def battery_margin_t(
    battery_mass_t: float,
    max_battery_mass_t: float,
) -> float:
    """Return positive margin when the battery mass is feasible.

    This arithmetic transform can be lifted over Bilt domains. A Boolean
    feasibility mask is constructed after evaluation because BiltTensor does
    not currently implement comparison operators.
    """

    return max_battery_mass_t - battery_mass_t


mission_model = Implementation(
    Mission,
    [
        gross_mass_t,
        shaft_energy_kwh,
        electrical_energy_kwh,
        battery_mass_t,
        battery_margin_t,
    ],
)


def as_array(value: object) -> np.ndarray:
    """Convert the current Bilt tensor representation into a NumPy array."""

    values = getattr(value, "values", value)
    return np.asarray(values)


def main() -> None:
    # Dimension identity is explicit and independent of array-axis position.
    scramble = Dim("scramble")
    sample = Dim("sample")
    route = Dim("route")
    payload_case = Dim("payload_case")

    scrambles = 8
    power = 10
    samples = 2**power

    draws = scrambled_sobol_inputs(
        scrambles=scrambles,
        power=power,
    )

    mission = Mission(
        # These three tensors share the (scramble, sample) domain.
        specific_energy_kwh_per_t=tensor(
            draws.specific_energy_kwh_per_t,
            over=(scramble, sample),
        ),
        propulsive_efficiency=tensor(
            draws.propulsive_efficiency,
            over=(scramble, sample),
        ),
        energy_multiplier=tensor(
            draws.energy_multiplier,
            over=(scramble, sample),
        ),

        # Each route is combined with every QMC sample and payload case.
        distance_km=tensor(
            [150.0, 250.0, 350.0, 450.0, 550.0, 650.0],
            over=route,
        ),

        # Each payload case is combined with every sample and route.
        payload_t=tensor(
            [2.0, 4.0, 6.0, 8.0],
            over=payload_case,
        ),

        # Scalars broadcast over every inherited domain.
        empty_mass_t=20.0,
        energy_per_km_t=0.18,
        usable_fraction=0.85,
        max_battery_mass_t=7.5,
    )

    backend = PythonBackend()

    with backend, mission_model:
        battery_mass_values = as_array(mission.battery_mass_t)
        battery_margin_values = as_array(mission.battery_margin_t)

    # Parameter order in the lifted transforms gives:
    #
    #   (scramble, sample, route, payload_case)
    #
    expected_shape = (
        scrambles,
        samples,
        6,
        4,
    )

    assert battery_mass_values.shape == expected_shape
    assert battery_margin_values.shape == expected_shape

    # Comparison is applied to the already-broadcast numeric tensor.
    feasible_values = battery_margin_values >= 0.0

    assert feasible_values.shape == expected_shape

    # First eliminate the sample domain independently within each scramble.
    #
    # Input:
    #   (scramble, sample, route, payload_case)
    #
    # Output:
    #   (scramble, route, payload_case)
    probability_by_scramble = feasible_values.mean(axis=1)

    # Then combine the independent randomized QMC estimates.
    #
    # Output:
    #   (route, payload_case)
    probability = probability_by_scramble.mean(axis=0)

    standard_error = (
        probability_by_scramble.std(axis=0, ddof=1)
        / np.sqrt(scrambles)
    )

    print(
        "Battery mass shape:",
        battery_mass_values.shape,
    )
    print(
        "Battery margin shape:",
        battery_margin_values.shape,
    )
    print(
        "Feasibility shape:",
        feasible_values.shape,
    )
    print(
        "Probability shape (route, payload_case):",
        probability.shape,
    )
    print(
        "450 km, 6 t probability:",
        probability[3, 2],
    )
    print(
        "450 km, 6 t standard error:",
        standard_error[3, 2],
    )


if __name__ == "__main__":
    main()