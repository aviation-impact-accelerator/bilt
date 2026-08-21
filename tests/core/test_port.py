import dataclasses

import pytest

from bilt.core._port import DOTTED_SEPARATOR, QualifiedPort, mangle_inputs, mangle_name


def test_dotted_joins_the_sub_system_and_the_port() -> None:
    assert QualifiedPort("resistor", "v").dotted == "resistor.v"


def test_str_is_the_dotted_spelling() -> None:
    assert str(QualifiedPort("resistor", "v")) == "resistor.v"


def test_the_separator_is_the_one_used_by_dotted() -> None:
    assert QualifiedPort("resistor", "v").dotted == f"resistor{DOTTED_SEPARATOR}v"


def test_qualified_ports_are_frozen() -> None:
    port = QualifiedPort("resistor", "v")

    with pytest.raises(dataclasses.FrozenInstanceError):
        port.port = "i"  # pyrefly: ignore[read-only]  # ty: ignore[invalid-assignment]


def test_qualified_ports_compare_and_hash_by_value() -> None:
    assert QualifiedPort("resistor", "v") == QualifiedPort("resistor", "v")
    assert len({QualifiedPort("resistor", "v"), QualifiedPort("resistor", "v")}) == 1
    assert QualifiedPort("resistor", "v") != QualifiedPort("capacitor", "v")


def test_unqualify_strips_the_sub_system_prefix() -> None:
    assert QualifiedPort.unqualify("resistor.v", "resistor", {"v", "i", "r"}) == "v"


def test_unqualify_rejects_a_name_belonging_to_another_sub_system() -> None:
    assert QualifiedPort.unqualify("capacitor.v", "resistor", {"v", "i", "r"}) is None


def test_unqualify_rejects_an_unqualified_name() -> None:
    assert QualifiedPort.unqualify("v", "resistor", {"v", "i", "r"}) is None


def test_unqualify_rejects_a_port_the_sub_system_does_not_have() -> None:
    assert QualifiedPort.unqualify("resistor.typo", "resistor", {"v", "i", "r"}) is None


def test_unqualify_keeps_the_remaining_qualification_of_a_nested_port() -> None:
    """A grandchild's port reaches the grandparent doubly qualified, and only the outer level is stripped here."""
    assert QualifiedPort.unqualify("outer.inner.v", "outer", {"inner.v"}) == "inner.v"


def test_unqualify_rejects_a_prefix_that_only_looks_like_the_sub_system() -> None:
    assert QualifiedPort.unqualify("resistor_bank.v", "resistor", {"v"}) is None


def test_mangle_inputs_leaves_a_plain_name_alone() -> None:
    assert mangle_inputs({"voltage": 1.0}) == {"voltage": 1.0}


def test_mangle_inputs_replaces_the_dotted_separator() -> None:
    assert mangle_inputs({"resistor.v": 1.0}) == {"resistor__v": 1.0}


def test_mangle_inputs_replaces_every_level_of_a_nested_name() -> None:
    assert mangle_inputs({"board.resistor.v": 1.0}) == {"board__resistor__v": 1.0}


def test_mangle_inputs_carries_every_value_across() -> None:
    assert mangle_inputs({"resistor.i": 2.0, "r": 3.0}) == {"resistor__i": 2.0, "r": 3.0}


def test_mangle_name_leaves_a_plain_name_alone() -> None:
    assert mangle_name("voltage") == "voltage"


def test_mangle_name_replaces_every_dotted_separator() -> None:
    assert mangle_name("board.resistor.v") == "board__resistor__v"
