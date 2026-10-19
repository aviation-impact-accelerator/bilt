"""A model with no dimensionality annotations evaluates exactly as it did before."""

import pytest

from bilt import PYTHON_BACKEND, Implementation, Interface, equation, field, system

VOLTS = 6.0
AMPS = 2.0
OHMS = 3.0


@system
class Resistor(metaclass=Interface):
    v: float = field()
    i: float = field()
    r: float = field()


@equation
def ohms_law(v: float, i: float, r: float) -> bool:
    return v == i * r


@pytest.mark.parametrize(
    ("target", "expected"),
    (("v", VOLTS), ("i", AMPS), ("r", OHMS)),
)
def test_unannotated_model_returns_a_bare_float(target: str, expected: float) -> None:
    known: dict[str, object] = {"v": VOLTS, "i": AMPS, "r": OHMS}
    del known[target]

    result = Implementation(Resistor, [ohms_law]).evaluate(known, target, PYTHON_BACKEND)

    assert result == expected
    assert isinstance(result, float)
