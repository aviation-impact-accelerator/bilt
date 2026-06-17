import pytest

from bilt.symbolic._atoms import (
    ADD,
    EQUALS,
    GREATER_THAN,
    GREATER_THAN_OR_EQUALS,
    INV,
    LESS_THAN,
    LESS_THAN_OR_EQUALS,
    MUL,
    NEG,
    NOT_EQUALS,
    POW,
    Atom,
    AtomType,
    FloatType,
    IntType,
    OpType,
    RelationOpType,
    SymbolType,
    atom_cache_size,
    clear_atom_cache,
    floating,
    integer,
    relation_operator,
    symbol,
)
from bilt.symbolic._errors import AtomMutationError


@pytest.fixture(autouse=True)
def clear_atoms_between_tests() -> None:
    clear_atom_cache()


def test_atom_type_constructs_atom() -> None:
    atom_type = AtomType("Custom")

    atom = atom_type("x")

    assert isinstance(atom, Atom)
    assert atom.atom_type is atom_type
    assert atom.value == "x"


def test_atoms_are_hash_consed_by_type_and_value() -> None:
    atom_type = AtomType("Custom")
    other_type = AtomType("Custom")

    x1 = atom_type("x")
    x2 = atom_type("x")
    y = atom_type("y")
    other_x = other_type("x")

    assert x1 is x2
    assert x1 is not y
    assert x1 is not other_x


def test_atom_equality_and_hash_are_identity_based() -> None:
    atom_type = AtomType("Custom")

    x = atom_type("x")
    x_again = atom_type("x")
    y = atom_type("y")

    assert x is x_again
    assert x == x_again
    assert x != y
    assert x != object()
    assert hash(x) == id(x)


def test_atom_is_immutable() -> None:
    x = symbol("x")

    with pytest.raises(AtomMutationError):
        x.value = "y"

    with pytest.raises(AtomMutationError):
        x.atom_type = OpType

    with pytest.raises(AtomMutationError):
        del x.value

    with pytest.raises(AtomMutationError):
        x.extra = "not allowed"  # type: ignore[attr-defined]


def test_symbol_integer_and_float_helpers_construct_expected_atoms() -> None:
    integer_value = 1
    float_value = 1.5

    x = symbol("x")
    one = integer(integer_value)
    const = floating(float_value)

    assert x.atom_type is SymbolType
    assert x.value == "x"

    assert one.atom_type is IntType
    assert one.value == integer_value

    assert const.atom_type is FloatType
    assert const.value == float_value


def test_symbol_integer_and_float_helpers_are_hash_consed() -> None:
    integer_value = 1
    float_value = 1.5

    assert symbol("x") is symbol("x")
    assert integer(integer_value) is integer(integer_value)
    assert floating(float_value) is floating(float_value)


@pytest.mark.parametrize(
    ("op", "value"),
    (
        (ADD, "add"),
        (MUL, "mul"),
        (POW, "pow"),
        (NEG, "neg"),
        (INV, "inv"),
    ),
)
def test_operation_constants_are_op_atoms(op: Atom, value: str) -> None:
    assert op.atom_type is OpType
    assert op.value == value


@pytest.mark.parametrize(
    ("op", "value"),
    (
        (EQUALS, "=="),
        (NOT_EQUALS, "!="),
        (LESS_THAN, "<"),
        (LESS_THAN_OR_EQUALS, "<="),
        (GREATER_THAN, ">"),
        (GREATER_THAN_OR_EQUALS, ">="),
    ),
)
def test_relation_operator_constants_are_relation_op_atoms(
    op: Atom,
    value: str,
) -> None:
    assert op.atom_type is RelationOpType
    assert op.value == value


@pytest.mark.parametrize(
    "value",
    (
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
    ),
)
def test_relation_operator_constructs_interned_relation_op_atom(value: str) -> None:
    op1 = relation_operator(value)
    op2 = relation_operator(value)

    assert op1 is op2
    assert op1.atom_type is RelationOpType
    assert op1.value == value


@pytest.mark.parametrize(
    "value",
    (
        "===",
        "<>",
        "=>",
        "=<",
        "contains",
        "",
    ),
)
def test_relation_operator_rejects_unknown_operator(value: str) -> None:
    with pytest.raises(ValueError, match="Unknown relation operator"):
        relation_operator(value)


def test_relation_operator_atom_is_distinct_from_plain_op_atom() -> None:
    relation_equals = relation_operator("==")
    plain_equals = Atom(OpType, "==")

    assert relation_equals is not plain_equals
    assert relation_equals != plain_equals
    assert relation_equals.atom_type is RelationOpType
    assert plain_equals.atom_type is OpType


def test_atom_cache_can_be_cleared() -> None:
    expected_cache_size_before_clear = 1
    expected_cache_size_after_clear = 0

    x = symbol("x")

    assert atom_cache_size() == expected_cache_size_before_clear

    clear_atom_cache()

    assert atom_cache_size() == expected_cache_size_after_clear

    # Keep the atom alive until after the cache assertion.
    assert x
