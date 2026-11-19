"""Wheatstone bridge: forward (currents) and inverse (balancing arm) solves.

Expected values are exact. With 10 V across arms of 100, 200, 150 and 250 ohm, a
50 ohm galvanometer, and node C grounded, nodal analysis gives node potentials
V_B = 660/101 V and V_D = 650/101 V. So the galvanometer carries
(V_B - V_D) / 50 ohm = 1/505 A, and the supply delivers
(10 V - V_B) / 100 ohm + (10 V - V_D) / 150 ohm = 59/1010 A. The bridge balances when
arm4 = arm2 * arm3 / arm1 = 300 ohm.

`examples.circuits` uses proposed APIs, so importing it raises
`ProposalNotImplementedError`. Each test imports it locally, so that the error marks
that test as an expected failure instead of stopping collection.
"""

import pytest
from bilt_proposals import ProposalNotImplementedError, Scope

from bilt.units import ampere, ohm, volt


@pytest.mark.xfail(
    raises=ProposalNotImplementedError, reason="connectors and choosing implementations inside systems are proposed"
)
def test_forward_solve() -> None:
    from examples.circuits import (  # noqa: PLC0415
        Ground,
        Resistor,
        VoltageSource,
        WheatstoneBridge,
        ideal_ground,
        ideal_resistor,
        ideal_voltage_source,
        ideal_wheatstone_bridge,
    )

    bridge = WheatstoneBridge(
        source=VoltageSource(voltage=10.0 * volt),
        ground=Ground(potential=0.0 * volt),
        arm1=Resistor(resistance=100.0 * ohm),
        arm2=Resistor(resistance=200.0 * ohm),
        arm3=Resistor(resistance=150.0 * ohm),
        arm4=Resistor(resistance=250.0 * ohm),
        galvanometer=Resistor(resistance=50.0 * ohm),
    )

    with ideal_wheatstone_bridge, ideal_resistor, ideal_voltage_source, ideal_ground:
        assert bridge.galvanometer.current.convert_to(ampere).value == pytest.approx(1 / 505)
        assert bridge.source.current.convert_to(ampere).value == pytest.approx(-59 / 1010)


@pytest.mark.xfail(
    raises=ProposalNotImplementedError, reason="connectors and choosing implementations inside systems are proposed"
)
def test_balancing_is_an_inverse_solve() -> None:
    from examples.circuits import (  # noqa: PLC0415
        Ground,
        Resistor,
        VoltageSource,
        WheatstoneBridge,
        ideal_ground,
        ideal_resistor,
        ideal_voltage_source,
        ideal_wheatstone_bridge,
    )

    bridge = WheatstoneBridge(
        source=VoltageSource(voltage=10.0 * volt),
        ground=Ground(potential=0.0 * volt),
        arm1=Resistor(resistance=100.0 * ohm),
        arm2=Resistor(resistance=200.0 * ohm),
        arm3=Resistor(resistance=150.0 * ohm),
        galvanometer=Resistor(resistance=50.0 * ohm, current=0.0 * ampere),
    )

    with Scope(ideal_wheatstone_bridge, ideal_resistor, ideal_voltage_source, ideal_ground):
        assert bridge.arm4.resistance.convert_to(ohm).value == pytest.approx(300.0)
