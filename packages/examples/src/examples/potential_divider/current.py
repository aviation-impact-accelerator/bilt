"""
This file shows implementation of a potential divider model as per my current
understanding of how the Bilt system is being implemented and will work.

       -----------------------
       |                     |   +
       |                    ---
       |                    | |   R_1
   +   |                    ---  -
  V   ----                   |__________ V_div
       --                    |
   -   |                    ---  +
       |                    | |   R_2
       |                    ---  -
       |                     |
       -----------------------
                 | G
                ---
                 -    0V

"""


from bilt.core import *
from bilt.core._equation import *

@system
class Pin(metaclass=Interface):
    """
    Note: Current `i` out of pin is +ve.
    """
    v: float = field()
    i: float = field()

@system
class SISO(metaclass=Interface):
    p: Pin
    n: Pin
    v: float = field()
    i: float = field()

@system
class Resistor(metaclass=Interface):
    siso: SISO
    R: float = field()

@system
class VoltageSource(metaclass=Interface):
    siso: SISO
    V: float = field()

@system
class Ground(metaclass=Interface):
    g: Pin

@equation
def connect_voltage(a: Pin, b: Pin) -> bool:
    return a.v == b.v

@equation
def connect_current(a: Pin, b: Pin) -> bool:
    return a.i + b.i == 0

def connect(a: Pin, b: Pin) -> list[bool]:  # We'd want type hinting to be static
    return [connect_voltage(a, b), connect_current(a, b)]

@equation
def siso_potential_difference(p: Pin, n: Pin, v: float) -> bool:
    return v == p.v - n.v

@equation
def siso_current_flow_constraint(p: Pin, n: Pin) -> bool:
    return p.i == -n.i  # I think? Depending on sign convention

@equation
def siso_current(p: Pin, n: Pin, i: float) -> bool:
    return i == p.i

@equation
def export_voltage_source_v(V: float, voltage_src: VoltageSource) -> bool:
    return V == voltage_src.siso.v

@equation
def export_r_1_resistance(R_1: float, r_1: Resistor) -> bool:
    return R_1 == r_1.R

@equation
def export_r_2_resistance(R_2: float, r_2: Resistor) -> bool:
    return R_2 == r_2.R

@equation
def export_V_div(V_div: float, r_1: Resistor) -> bool:
    return V_div == r_1.siso.n.v

@system
class WholeCircuit(metaclass=Interface):
    V: float = field()
    R_1: float = field()
    R_2: float = field()
    V_div: float = field()
    voltage_src: VoltageSource
    r_1: Resistor
    r_2: Resistor
    gnd: Ground

voltage_src = VoltageSource(siso=SISO(p=Pin(), n=Pin()))
r_1 = Resistor(siso=SISO(p=Pin(), n=Pin()))
r_2 = Resistor(siso=SISO(p=Pin(), n=Pin()))
gnd = Ground(g=Pin())
whole_circuit = WholeCircuit(
    V=5.0, R_1=1000, R_2=2000,
    voltage_src=voltage_src, r_1=r_1, r_2=r_2, gnd=gnd)

eqns = [
    *connect(voltage_src.p, r_1.p),
    *connect(r_1.n, r_2.p),
    *connect(r_2.n, voltage_src.n),
    *connect(voltage_src.n, gnd.g),
    export_voltage_source_v,
    export_r_1_resistance,
    export_r_2_resistance,
    export_V_div
]

impl = Implementation(WholeCircuit, eqns)

with impl:
    print(whole_circuit.V_div)
