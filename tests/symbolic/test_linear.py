import pytest

from bilt.core._equation import equation
from bilt.symbolic._atoms import NEG
from bilt.symbolic._isolation._expr_ops import same_tree
from bilt.symbolic._linear import LinearisationError, linearise
from bilt.symbolic._tree import Tree, int_tree, symbol_tree


def test_linearise_coupled_equations() -> None:
    @equation
    def first(x: float, y: float, a: float) -> bool:
        return x + y == a

    @equation
    def second(x: float, y: float, b: float) -> bool:
        return x - y == b

    system = linearise(
        (
            first.__bilt_equation_tree__,
            second.__bilt_equation_tree__,
        ),
        ("x", "y"),
    )

    assert system.unknowns == ("x", "y")

    assert same_tree(
        system.coefficients[0][0],
        int_tree(1),
    )
    assert same_tree(
        system.coefficients[0][1],
        int_tree(1),
    )
    assert same_tree(
        system.coefficients[1][0],
        int_tree(1),
    )
    assert same_tree(
        system.coefficients[1][1],
        Tree(NEG, (int_tree(1),)),
    )

    assert same_tree(
        system.rhs[0],
        symbol_tree("a"),
    )
    assert same_tree(
        system.rhs[1],
        symbol_tree("b"),
    )


def test_linearise_allows_symbolic_coefficients() -> None:
    @equation
    def relation(
        x: float,
        y: float,
        k: float,
        a: float,
    ) -> bool:
        return k * x + y == a

    system = linearise(
        (relation.__bilt_equation_tree__,),
        ("x",),
    )

    assert same_tree(
        system.coefficients[0][0],
        symbol_tree("k"),
    )


def test_linearise_rejects_nonlinear_unknown() -> None:
    @equation
    def relation(
        x: float,
        a: float,
    ) -> bool:
        return x * x == a

    with pytest.raises(LinearisationError):
        linearise(
            (relation.__bilt_equation_tree__,),
            ("x",),
        )
