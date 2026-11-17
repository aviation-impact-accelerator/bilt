"""Wheatstone bridge: forward (currents) and inverse (balancing arm) solves.

Expected values are exact. With 10 V across arms of 100, 200, 150 and 250 ohm, a
50 ohm galvanometer, and node C grounded, nodal analysis gives node potentials
V_B = 660/101 V and V_D = 650/101 V. So the galvanometer carries
(V_B - V_D) / 50 ohm = 1/505 A, and the supply delivers
(10 V - V_B) / 100 ohm + (10 V - V_D) / 150 ohm = 59/1010 A. The bridge balances when
arm4 = arm2 * arm3 / arm1 = 300 ohm.
"""

import pytest
from bilt_proposals import ProposalNotImplementedError
from examples.circuits import WheatstoneBridge, wheatstone_bridge

from bilt.units import ampere, ohm, volt


def test_forward_solve() -> None:
    bridge = WheatstoneBridge(
        supply_voltage=10.0 * volt,
        ground_potential=0.0 * volt,
        arm1_resistance=100.0 * ohm,
        arm2_resistance=200.0 * ohm,
        arm3_resistance=150.0 * ohm,
        arm4_resistance=250.0 * ohm,
        galvanometer_resistance=50.0 * ohm,
    )

    with wheatstone_bridge:
        assert bridge.galvanometer_current.convert_to(ampere).value == pytest.approx(1 / 505)
        assert bridge.supply_current.convert_to(ampere).value == pytest.approx(59 / 1010)


def test_balancing_is_an_inverse_solve() -> None:
    bridge = WheatstoneBridge(
        supply_voltage=10.0 * volt,
        ground_potential=0.0 * volt,
        arm1_resistance=100.0 * ohm,
        arm2_resistance=200.0 * ohm,
        arm3_resistance=150.0 * ohm,
        galvanometer_resistance=50.0 * ohm,
        galvanometer_current=0.0 * ampere,
    )

    with wheatstone_bridge:
        assert bridge.arm4_resistance.convert_to(ohm).value == pytest.approx(300.0)


@pytest.mark.xfail(raises=ProposalNotImplementedError, reason="connectors are proposed")
def test_forward_solve_with_proposed_connectors() -> None:
    from examples.circuits import proposed  # noqa: PLC0415

    bridge = proposed.WheatstoneBridge(
        source=proposed.VoltageSource(voltage=10.0 * volt),
        ground=proposed.Ground(potential=0.0 * volt),
        arm1=proposed.Resistor(resistance=100.0 * ohm),
        arm2=proposed.Resistor(resistance=200.0 * ohm),
        arm3=proposed.Resistor(resistance=150.0 * ohm),
        arm4=proposed.Resistor(resistance=250.0 * ohm),
        galvanometer=proposed.Resistor(resistance=50.0 * ohm),
    )

    with proposed.wheatstone_bridge:
        assert bridge.galvanometer.current.convert_to(ampere).value == pytest.approx(1 / 505)
        # Passive sign convention: a source delivering power carries negative current.
        assert bridge.source.current.convert_to(ampere).value == pytest.approx(-59 / 1010)


@pytest.mark.xfail(raises=ProposalNotImplementedError, reason="connectors are proposed")
def test_balancing_is_an_inverse_solve_with_proposed_connectors() -> None:
    from examples.circuits import proposed  # noqa: PLC0415

    bridge = proposed.WheatstoneBridge(
        source=proposed.VoltageSource(voltage=10.0 * volt),
        ground=proposed.Ground(potential=0.0 * volt),
        arm1=proposed.Resistor(resistance=100.0 * ohm),
        arm2=proposed.Resistor(resistance=200.0 * ohm),
        arm3=proposed.Resistor(resistance=150.0 * ohm),
        galvanometer=proposed.Resistor(resistance=50.0 * ohm, current=0.0 * ampere),
    )

    with proposed.wheatstone_bridge:
        assert bridge.arm4.resistance.convert_to(ohm).value == pytest.approx(300.0)
