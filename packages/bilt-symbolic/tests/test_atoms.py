# tests/symbolic/test_atoms.py

import pytest

from bilt.symbolic._atoms import Atom, AtomType, atom_cache_size, clear_atom_cache
from bilt.symbolic._errors import AtomHashError, AtomMutationError


@pytest.fixture(autouse=True)
def clear_atoms_between_tests() -> None:
    clear_atom_cache()


def test_atom_type_constructs_atom() -> None:
    symbol = AtomType("symbol")

    atom = symbol("x")

    assert isinstance(atom, Atom)
    assert atom.atom_type is symbol
    assert atom.value == "x"


def test_same_atom_type_and_value_returns_same_object() -> None:
    symbol = AtomType("symbol")

    x1 = symbol("x")
    x2 = symbol("x")

    assert x1 is x2
    assert x1 == x2


def test_same_atom_type_different_value_returns_different_object() -> None:
    symbol = AtomType("symbol")

    x = symbol("x")
    y = symbol("y")

    assert x is not y
    assert x != y


def test_different_atom_type_same_value_returns_different_object() -> None:
    symbol = AtomType("symbol")
    op = AtomType("Op")

    symbol_x = symbol("x")
    op_x = op("x")

    assert symbol_x is not op_x
    assert symbol_x != op_x


def test_atom_types_with_same_name_are_distinct() -> None:
    symbol1 = AtomType("symbol")
    symbol2 = AtomType("symbol")

    x1 = symbol1("x")
    x2 = symbol2("x")

    assert symbol1 is not symbol2
    assert x1 is not x2
    assert x1 != x2


def test_atom_value_cannot_be_mutated() -> None:
    symbol = AtomType("symbol")
    x = symbol("x")

    with pytest.raises(AtomMutationError):
        x.value = "y"


def test_atom_type_cannot_be_mutated() -> None:
    symbol = AtomType("symbol")
    op = AtomType("op")
    x = symbol("x")

    with pytest.raises(AtomMutationError):
        x.atom_type = op


def test_new_attributes_cannot_be_added() -> None:
    symbol = AtomType("symbol")
    x = symbol("x")

    with pytest.raises(AtomMutationError):
        x.extra = "not allowed"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "value",
    (
        ["x"],
        {"x": 1},
        {"x", "y"},
    ),
)
def test_unhashable_values_raise_atom_hash_error(value: object) -> None:
    symbol = AtomType("symbol")

    with pytest.raises(AtomHashError):
        symbol(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    (
        "x",
        1,
        1.5,
        True,
        ("x", 1),
        frozenset({"x", "y"}),
    ),
)
def test_hashable_values_are_accepted(value: object) -> None:
    symbol = AtomType("symbol")

    atom = symbol(value)  # type: ignore[arg-type]

    assert atom.value == value


def test_equality_is_identity_based() -> None:
    symbol = AtomType("symbol")

    x1 = symbol("x")
    x2 = symbol("x")
    y = symbol("y")

    assert x1 is x2
    assert x1 == x2
    assert x1 is not y
    assert x1 != y


def test_hash_is_identity_based_for_interned_atoms() -> None:
    symbol = AtomType("symbol")

    x1 = symbol("x")
    x2 = symbol("x")

    assert x1 is x2
    assert hash(x1) == hash(x2)


def test_repr_uses_atom_type_and_value() -> None:
    symbol = AtomType("symbol")

    x = symbol("x")

    assert repr(x) == "symbol('x')"


def test_str_uses_value_string() -> None:
    symbol = AtomType("symbol")

    x = symbol("x")

    assert str(x) == "x"


def test_atom_cache_size_counts_live_atoms() -> None:
    symbol = AtomType("symbol")

    x = symbol("x")
    y = symbol("y")
    x_again = symbol("x")

    expected_size = 2

    assert x is x_again
    assert x is not y
    assert atom_cache_size() == expected_size


def test_clear_atom_cache_clears_cache() -> None:
    symbol = AtomType("symbol")

    x = symbol("x")

    assert atom_cache_size() == 1

    clear_atom_cache()

    assert atom_cache_size() == 0

    # Keep the atom alive until after the assertion.
    assert x


def test_clear_atom_cache_allows_new_interning() -> None:
    symbol = AtomType("symbol")

    x1 = symbol("x")

    clear_atom_cache()

    x2 = symbol("x")
    x3 = symbol("x")

    assert x1 is not x2
    assert x1 != x2

    assert x2 is x3
    assert x2 == x3


def test_atom_value_cannot_be_deleted() -> None:
    symbol = AtomType("symbol")
    x = symbol("x")

    with pytest.raises(AtomMutationError):
        del x.value


def test_atom_type_cannot_be_deleted() -> None:
    symbol = AtomType("symbol")
    x = symbol("x")

    with pytest.raises(AtomMutationError):
        del x.atom_type


def test_unknown_attribute_cannot_be_deleted() -> None:
    symbol = AtomType("symbol")
    x = symbol("x")

    with pytest.raises(AtomMutationError):
        delattr(x, "extra")
