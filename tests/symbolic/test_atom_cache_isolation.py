from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

pytest_plugins = ("pytester",)


def test_atom_tests_preserve_collected_equations(pytester: pytest.Pytester) -> None:
    root = Path(__file__).resolve().parents[2]
    pytester.makeini(f"[pytest]\npythonpath = {root / 'src'}\n")
    pytester.makepyfile(test_atoms=Path(__file__).with_name("test_atoms.py").read_text())
    pytester.makepyfile(
        test_retained_equation="""
from bilt.symbolic._atoms import ADD, symbol
from bilt.symbolic._expressions import Expr
from bilt.symbolic._isolation._isolate import isolate

x = Expr.symbol("x")
a = Expr.symbol("a")
b = Expr.symbol("b")
equation = x == a + b
original_x = symbol("x")

def test_collected_equation_survives_atom_tests():
    assert symbol("x") is original_x
    assert isolate(equation, Expr.symbol("x")).rhs is not None
    assert ADD.atom_type("add") is ADD
""",
    )
    result = pytester.runpytest_subprocess("test_atoms.py", "test_retained_equation.py", "-q")
    result.assert_outcomes(passed=35)
