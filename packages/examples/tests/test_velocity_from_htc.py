from __future__ import annotations
import pytest
from examples.symbolic_api_and_basic_backend.examples import build_velocity_from_htc_model

def test_solves_velocity_from_desired_heat_transfer_coefficient() -> None:
    model = build_velocity_from_htc_model()
    result = model.solve(inputs={
        "heat_transfer_coefficient": 4754.327547388351,
        "density": 1_000.0,
        "length": 0.01,
        "dynamic_viscosity": 0.001,
        "specific_heat_capacity": 4_180.0,
        "thermal_conductivity": 0.6,
        "exponent": 0.4,
    }, outputs=["velocity"])
    assert result.requested_outputs == {"velocity": pytest.approx(1.0)}
    assert result.all_values["prandtl_number"] == pytest.approx(6.966666666666667)
    assert result.all_values["nusselt_number"] == pytest.approx(79.23879245647252)
    assert result.all_values["reynolds_number"] == pytest.approx(10_000.0)
    assert [t.outputs for t in result.graph.transforms] == [("prandtl_number",), ("nusselt_number",), ("reynolds_number",), ("velocity",)]
    assert result.graph.dependencies() == {
        "prandtl_number": {"dynamic_viscosity", "specific_heat_capacity", "thermal_conductivity"},
        "nusselt_number": {"heat_transfer_coefficient", "length", "thermal_conductivity"},
        "reynolds_number": {"exponent", "nusselt_number", "prandtl_number"},
        "velocity": {"density", "dynamic_viscosity", "length", "reynolds_number"},
    }

def test_model_unit_analysis_reports_equations() -> None:
    report = build_velocity_from_htc_model().unit_report()
    assert "Unit analysis for model 'velocity_from_desired_htc'" in report
    assert "reynolds_number_definition: OK" in report
    assert "prandtl_number_definition: OK" in report
    assert "nusselt_number_definition: OK" in report
    assert "heat_transfer_coefficient_definition: OK" in report
