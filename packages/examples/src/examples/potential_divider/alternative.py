"""
This file shows implementation of a potential divider model with an alternative
workflow to what I understand as being currently implemented in Bilt. This is
largely an exercise for me to learn the what design decisions have already been made
and why.

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


import abc
import dataclasses
import typing


"""
General infrastructure.
"""

@dataclasses.dataclass
class Tree[T]:    
    value: T
    children: list[T]

class TreeType[T](typing.Protocol):
    def _tree_rep(self) -> Tree[T]: ...

class Expr:
    def __init__(self, children: list[Expr] = []):
        self._tree = Tree[Expr](self, children.copy())


    def __eq__(self, value: Expr | float | int) -> Equation:
        ...


    def __mul__(self, other: Expr | float | int) -> Expr:
        ...
   
    def __rmul__(self, other: Expr | float | int) -> Expr:
        ...
   
    def __neg__(self) -> Expr:
        ...
   
    def __add__(self, other: Expr | float | int) -> Expr:
        ...


    def __radd__(self, other: Expr | float | int) -> Expr:
        ...
   
    def __sub__(self, other: Expr | float | int) -> Expr:
        ...


    def __rsub__(self, other: Expr | float | int) -> Expr:
        ...
   
    def __pow__(self, other: Expr | float | int) -> Expr:
        ...
   
    def __rpow__(self, other: Expr | float | int) -> Expr:
        ...
   
    def _tree_rep(self) -> Tree[Expr]:
        return self._tree

class Float(Expr):
    def __init__(self, value: float | None = None) -> None:
        super().__init__()


        self._value = value

class System(abc.ABC):
    def __init__(self):
        super().__init__()
        self.setup()

    @abc.abstractmethod
    def setup(self) -> None: ...

    def add_equations(self, *args: Equation | list[Equation]) -> None:
        if self._equations is None:
            self._equations: list[Equation] | None = []

        for arg in args:
            match arg:
                case Equation():
                    self._equations.append(arg)
                case list():
                    self._equations.extend(arg)

class Equals(Expr): ...

class Equation:
    """ A relational, two-child tree. For now, equation is specifically
     an equality. """
    def __init__(self, left: Expr, right: Expr) -> None:
        self._tree = Tree[Expr](Equals(), [left, right])

    @property
    def left(self) -> Expr:
        return self._tree.children[0]
   
    @property
    def right(self) -> Expr:
        return self._tree.children[1]
   
    def _tree_rep(self) -> Tree[Expr]:
        return self._tree


"""
Electrical-domain infrastructure.
"""

class Pin:
    def __init__(self):
        self.v = Float()
        self.i = Float()

class SISO(System):
    def setup(self):
        self.p = Pin()
        self.n = Pin()
        self.v = Float()
        self.i = Float()

        self.add_equations(
            self.v == self.p.v - self.n.v,
            self.p.i == -self.n.i,
            self.i == self.p.i
        )

class Resistor(SISO):
    def setup(self):
        self.R = Float()

        self.add_equations(
            self.v == self.i * self.R
        )

class VoltageSource(SISO):
    def setup(self):
        self.V = Float()

        self.add_equations(
            self.v == self.V
        )

class Ground(System):
    def setup(self):
        self.g = Pin()

        self.add_equations(
            self.g.v == 0
        )

def connect(a: Pin, b: Pin) -> list[Equation]:
    return [
        a.v == b.v,
        a.i == -b.i
    ]


"""
Specific model.
"""

class WholeCircuit(System):
    def setup(self):
        self.voltage_src = VoltageSource()
        self.r1 = Resistor()
        self.r2 = Resistor()
        self.gnd = Ground()

        self.V_div = Float()

        self.add_equations(
            connect(self.voltage_src.p, self.r1.p),
            connect(self.r1.n, self.r2.p),
            connect(self.r2.n, self.voltage_src.n),
            connect(self.voltage_src.n, self.gnd.g),
            self.V_div == self.r1.n.v
        )

circuit = WholeCircuit()
print(circuit.V_div)
