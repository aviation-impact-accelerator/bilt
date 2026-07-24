import pytest

from examples.brayton_model.example import example


def test_example() -> None:
    cycle = example()

    assert (
        cycle.net_specific_work,
        cycle.shaft_power,
        cycle.thermal_efficiency,
        cycle.exhaust_temperature,
        cycle.exhaust_pressure,
        cycle.waste_heat,
    ) == pytest.approx(
        (
            367_058.5154715981,
            7_341_170.309431962,
            0.3448987695293381,
            981.8728701775144,
            101_325.0,
            13_943_829.690568037,
        )
    )