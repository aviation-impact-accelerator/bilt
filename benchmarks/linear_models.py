"""Engineering fixtures for cold compilation and warm public linear evaluation."""

# ruff: noqa: INP001 - benchmark companion, not an installed package

import dataclasses
import typing
from contextlib import ExitStack
from unittest.mock import patch

import jax
import numpy as np

from bilt import Domain, Implementation, Interface, Quantity, Tensor, dim, domain, equation, field, system, tensor
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._system import _implementation as implementation_module
from bilt.units.dimensions import CURRENT, POWER, RESISTANCE, TEMPERATURE, VOLTAGE
from bilt.units.units import ampere, kelvin, ohm, volt, watt

if typing.TYPE_CHECKING:
    from bilt import Unit
    from bilt._backend import Backend, BackendCallable

COORDINATE = dim("branch")
Current = typing.Annotated[Tensor, CURRENT]
Voltage = typing.Annotated[Quantity, VOLTAGE]
Resistance = typing.Annotated[Quantity, RESISTANCE]
Temperature = typing.Annotated[Quantity, TEMPERATURE]
Heat = typing.Annotated[Tensor, POWER]
Conductance = typing.Annotated[Quantity, POWER / TEMPERATURE]
ThermalResistance = typing.Annotated[Quantity, TEMPERATURE / POWER]


@dataclasses.dataclass(frozen=True)
class LinearCase:
    """One physical network and layout, shared by both timing phases."""

    family: str
    shared: bool
    hinted: bool
    backend_name: str
    coordinates: int = 3
    depth: int = 0

    @property
    def label(self) -> str:
        """Stable parameter ID for saved comparisons."""
        return (
            f"{self.family}-{'shared' if self.shared else 'pointwise'}-"
            f"{'hinted' if self.hinted else 'inferred'}-{self.backend_name}-n{self.coordinates}-depth{self.depth}"
        )

    @property
    def backend(self) -> Backend:
        """Numerical backend without changing production defaults."""
        return PYTHON_BACKEND if self.backend_name == "python" else JAX_BACKEND


def linear_cases(*, extended: bool) -> tuple[LinearCase, ...]:
    """Keep small representatives routine; scale coordinates and depth independently."""
    sizes = ((3, 0), (16, 0), (3, 2)) if extended else ((3, 0),)
    return tuple(
        LinearCase(family, shared, hinted, backend, size, depth)
        for family in ("electrical", "thermal")
        for shared in (False, True)
        for hinted in (False, True)
        for backend in ("python", "jax")
        for size, depth in sizes
    )


@system
class ElectricalPorts(metaclass=Interface):
    """Branch current and terminal load voltage."""

    current: Current = field()
    voltage: Voltage = field()
    load: Resistance = field()


@system
class ElectricalSource(metaclass=Interface):
    """Source voltage and series resistance."""

    emf: Voltage = field()
    resistance: Resistance = field()


@equation
def electrical_supply(emf: Voltage, resistance: Resistance, inner: ElectricalPorts) -> bool:
    """Kirchhoff voltage balance across each source branch."""
    return emf == resistance * inner.current + inner.voltage


@system
class ThermalPorts(metaclass=Interface):
    """Two thermal nodes joined by a contact resistance."""

    hot: Temperature = field()
    cold: Temperature = field()
    heat: Heat = field()
    contact: Conductance = field()
    sink: ThermalResistance = field()


@system
class ThermalSource(metaclass=Interface):
    """Prescribed source temperature rise and series thermal resistance."""

    source: Temperature = field()
    resistance: ThermalResistance = field()


@equation
def thermal_supply(source: Temperature, resistance: ThermalResistance, inner: ThermalPorts) -> bool:
    """Thermal drop between source and hot contact node."""
    return source == inner.hot + resistance * inner.heat


@dataclasses.dataclass(frozen=True)
class Expected:
    """Independent physical solution in coherent units for one output."""

    values: np.ndarray
    unit: Unit
    domain: Domain

    def check(self, result: object) -> None:
        """Check units, named axes, shape and the independent analytic answer."""
        if self.domain.is_scalar:
            if not isinstance(result, Quantity):
                message = f"Expected a scalar Quantity, received {type(result)}."
                raise AssertionError(message)
            if result.unit != self.unit:
                message = f"Expected unit {self.unit}, received {result.unit}."
                raise AssertionError(message)
            values = np.asarray(result.value)
        else:
            if not isinstance(result, Tensor):
                message = f"Expected a Tensor, received {type(result)}."
                raise AssertionError(message)
            if result.domain != self.domain:
                message = f"Expected domain {self.domain}, received {result.domain}."
                raise AssertionError(message)
            if result.unit != self.unit:
                message = f"Expected unit {self.unit}, received {result.unit}."
                raise AssertionError(message)
            values = np.asarray(result.values)
        if values.shape != self.values.shape:
            message = f"Expected shape {self.values.shape}, received {values.shape}."
            raise AssertionError(message)
        np.testing.assert_allclose(values, self.values, rtol=2e-5, atol=2e-5)


def _electrical(case: LinearCase) -> tuple[Implementation, dict[str, object], dict[str, Expected]]:
    if typing.TYPE_CHECKING:
        current_type = Current
        voltage_type = Voltage
    else:
        current_type = typing.Annotated[Tensor, CURRENT, domain(COORDINATE)] if case.hinted else Current
        voltage_type = (
            typing.Annotated[Tensor, VOLTAGE, Domain.scalar() if case.shared else domain(COORDINATE)]
            if case.hinted
            else Voltage
        )

    @equation
    def load(current: current_type, voltage: voltage_type, load: Resistance) -> bool:
        return voltage == load * (current.sum(COORDINATE) if case.shared else current)

    @equation
    def expose_current(current: current_type, inner: ElectricalPorts) -> bool:
        return current == inner.current

    @equation
    def expose_voltage(voltage: voltage_type, inner: ElectricalPorts) -> bool:
        return voltage == inner.voltage

    @equation
    def supply_load(load: Resistance, inner: ElectricalPorts) -> bool:
        return inner.load == load

    child = Implementation(ElectricalPorts, [load])
    for _ in range(case.depth):
        child = Implementation(
            ElectricalPorts, [expose_current, expose_voltage, supply_load], internal_implementations={"inner": child}
        )
    model = Implementation(ElectricalSource, [electrical_supply], internal_implementations={"inner": child})
    emf = np.linspace(10.0, 14.0, case.coordinates)
    source_resistance = 1.0
    load_resistance = 0.5
    voltage = (
        load_resistance * np.sum(emf / source_resistance) / (1 + load_resistance * case.coordinates / source_resistance)
        if case.shared
        else emf * load_resistance / (source_resistance + load_resistance)
    )
    current = (emf - voltage) / source_resistance
    inputs: dict[str, object] = {
        "emf": tensor(emf, over=COORDINATE, unit=volt),
        "resistance": source_resistance * ohm,
        "inner.load": load_resistance * ohm,
    }
    expected = {
        "inner.current": Expected(current, ampere, domain(COORDINATE)),
        "inner.voltage": Expected(np.asarray(voltage), volt, Domain.scalar() if case.shared else domain(COORDINATE)),
    }
    return model, inputs, expected


def _thermal(case: LinearCase) -> tuple[Implementation, dict[str, object], dict[str, Expected]]:
    if typing.TYPE_CHECKING:
        hot_type = Temperature
        heat_type = Heat
        cold_type = Temperature
    else:
        hot_type = typing.Annotated[Tensor, TEMPERATURE, domain(COORDINATE)] if case.hinted else Temperature
        heat_type = typing.Annotated[Tensor, POWER, domain(COORDINATE)] if case.hinted else Heat
        cold_type = (
            typing.Annotated[Tensor, TEMPERATURE, Domain.scalar() if case.shared else domain(COORDINATE)]
            if case.hinted
            else Temperature
        )

    @equation
    def contact(hot: hot_type, cold: cold_type, heat: heat_type, contact: Conductance) -> bool:
        return heat == contact * (hot - cold)

    @equation
    def sink(cold: cold_type, heat: heat_type, sink: ThermalResistance) -> bool:
        return cold == sink * (heat.sum(COORDINATE) if case.shared else heat)

    @equation
    def expose_hot(hot: hot_type, inner: ThermalPorts) -> bool:
        return hot == inner.hot

    @equation
    def expose_cold(cold: cold_type, inner: ThermalPorts) -> bool:
        return cold == inner.cold

    @equation
    def expose_heat(heat: heat_type, inner: ThermalPorts) -> bool:
        return heat == inner.heat

    @equation
    def supply_contact(contact: Conductance, inner: ThermalPorts) -> bool:
        return inner.contact == contact

    @equation
    def supply_sink(sink: ThermalResistance, inner: ThermalPorts) -> bool:
        return inner.sink == sink

    child = Implementation(ThermalPorts, [contact, sink])
    for _ in range(case.depth):
        child = Implementation(
            ThermalPorts,
            [expose_hot, expose_cold, expose_heat, supply_contact, supply_sink],
            internal_implementations={"inner": child},
        )
    model = Implementation(ThermalSource, [thermal_supply], internal_implementations={"inner": child})
    # Temperatures are rises above a common ambient, not absolute temperatures.
    source = np.linspace(30.0, 50.0, case.coordinates)
    source_resistance = 0.5
    contact_conductance = 2.0
    sink_resistance = 0.25
    total_branch_resistance = source_resistance + 1 / contact_conductance
    cold = (
        sink_resistance
        * np.sum(source / total_branch_resistance)
        / (1 + sink_resistance * case.coordinates / total_branch_resistance)
        if case.shared
        else source * sink_resistance / (total_branch_resistance + sink_resistance)
    )
    heat = (source - cold) / total_branch_resistance
    hot = source - source_resistance * heat
    inputs: dict[str, object] = {
        "source": tensor(source, over=COORDINATE, unit=kelvin),
        "resistance": source_resistance * (kelvin / watt),
        "inner.contact": contact_conductance * (watt / kelvin),
        "inner.sink": sink_resistance * (kelvin / watt),
    }
    expected = {
        "inner.heat": Expected(heat, watt, domain(COORDINATE)),
        "inner.hot": Expected(hot, kelvin, domain(COORDINATE)),
        "inner.cold": Expected(np.asarray(cold), kelvin, Domain.scalar() if case.shared else domain(COORDINATE)),
    }
    return model, inputs, expected


@dataclasses.dataclass
class LinearTrial:
    """Fresh models for cold timing; precompiled, warmed models for evaluation."""

    case: LinearCase
    phase: str
    implementation: Implementation
    inputs: dict[str, object]
    expected: dict[str, Expected]
    request: EvaluationRequest
    resources: ExitStack
    result: object = None
    compiled: BackendCallable | None = None
    checked: bool = False
    closed: bool = False
    ran: bool = False

    @classmethod
    def prepare(cls, case: LinearCase, phase: str) -> LinearTrial:
        """Enter a fresh model/context and optionally compile and warm the target."""
        if phase not in {"analysis", "evaluation"}:
            raise ValueError(phase)
        resources = ExitStack()
        try:
            resources.enter_context(
                patch.object(implementation_module, "CausalizationStructuralAnalyzer", CausalizationStructuralAnalyzer)
            )
            model, inputs, expected = _electrical(case) if case.family == "electrical" else _thermal(case)
            context = resources.enter_context(EvaluationContext())
            context.domains.bind_all(inputs.values())
            request = EvaluationRequest(
                output=next(iter(expected)),
                input_domains={
                    name: value.domain if isinstance(value, Tensor) else Domain.scalar()
                    for name, value in inputs.items()
                },
            )
            trial = cls(case, phase, model, inputs, expected, request, resources)
            if phase == "evaluation":
                # Exclude Bilt compilation and initial JAX tracing/dispatch from warm timing.
                warm = model.evaluate(inputs, request.output, case.backend)
                _synchronize(warm, case.backend_name)
                expected[request.output].check(warm)
        except BaseException:
            resources.close()
            raise
        else:
            return trial

    def analyse(self) -> None:
        """Run exactly one timed operation in this trial."""
        if self.closed or self.ran:
            message = "Each linear benchmark trial must run once."
            raise RuntimeError(message)
        self.ran = True
        if self.phase == "analysis":
            self.compiled = self.implementation.get_evaluation(self.request, self.case.backend)
        else:
            self.result = self.implementation.evaluate(self.inputs, self.request.output, self.case.backend)
            _synchronize(self.result, self.case.backend_name)

    def check(self) -> dict[str, object]:
        """Check all physical outputs outside timing, including the timed result."""
        if not self.ran:
            message = "Cannot check a trial before running it."
            raise AssertionError(message)
        target = self.expected[self.request.output]
        if self.phase == "analysis":
            if self.compiled is None or self.compiled.output_domain != target.domain:
                message = "Compiled target has the wrong output domain."
                raise AssertionError(message)
        else:
            target.check(self.result)
        for output, expected in self.expected.items():
            value = self.implementation.evaluate(self.inputs, output, self.case.backend)
            _synchronize(value, self.case.backend_name)
            expected.check(value)
        self.checked = True
        return {
            "status": "success",
            "family": self.case.family,
            "layout": "shared" if self.case.shared else "pointwise",
            "domain_hints": self.case.hinted,
            "backend": self.case.backend_name,
            "phase": self.phase,
            "coordinates": self.case.coordinates,
            "wrapper_depth": self.case.depth,
            "target": self.request.output,
            "checked_outputs": list(self.expected),
            "physical_unknowns": (
                self.case.coordinates + 1 if self.case.family == "electrical" else 2 * self.case.coordinates + 1
            )
            if self.case.shared
            else (2 if self.case.family == "electrical" else 3) * self.case.coordinates,
            "expected_values": {name: value.values.tolist() for name, value in self.expected.items()},
            "expected_domains": {
                name: [str(dim) for dim in value.domain.dims] for name, value in self.expected.items()
            },
        }

    def close(self) -> None:
        """Restore solver selection and leave the evaluation context."""
        self.resources.close()
        self.closed = True


def _synchronize(value: object, backend_name: str) -> None:
    """Include device completion, including ordered validation callbacks, in timing."""
    if backend_name != "jax":
        return
    raw = value.values if isinstance(value, Tensor) else value.value if isinstance(value, Quantity) else value
    jax.block_until_ready(raw)
    jax.effects_barrier()
