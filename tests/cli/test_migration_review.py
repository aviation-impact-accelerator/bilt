"""Review diagnostics are advisory and do not invent scalar semantics."""

from bilt._migrate import migrate_source


def test_quantity_runtime_checks_are_reported_without_changes() -> None:
    source = (
        "from bilt import Quantity, QuantityTensor\n"
        "def extract(value):\n"
        "    assert isinstance(value, Quantity)\n"
        "    if not isinstance(value, QuantityTensor):\n"
        "        return value.value\n"
        "    return value.values\n"
    )
    result = migrate_source(source)
    assert result.source == source
    assert result.notes == (
        "3: review Quantity runtime check; it now also accepts quantity tensors",
        "4: review scalar assumptions; excluding QuantityTensor does not narrow Quantity.value",
    )
    assert migrate_source(result.source) == result


def test_scalar_magnitude_review_and_explicit_conversion() -> None:
    source = (
        "from bilt.units import year, second\n"
        "seconds = (1.0 * year).convert_to(second).value\n"
        "explicit = float((1.0 * year).convert_to(second).value)\n"
    )
    result = migrate_source(source)
    assert result.source == source
    assert result.notes == (
        "2: review scalar magnitude use; Quantity.value is Number, use float(...) if scalar is intended",
    )


def test_unrelated_and_shadowed_checks_are_not_reported() -> None:
    source = (
        "from other import Quantity\n"
        "from bilt import Quantity as Q\n"
        "assert isinstance(value, Quantity)\n"
        "def shadow(Q):\n"
        "    return isinstance(value, Q)\n"
    )
    assert not migrate_source(source).notes


def test_retained_import_report_does_not_prevent_rewrite() -> None:
    source = (
        "from bilt import Quantity, QuantityTensor\nMass = Quantity | QuantityTensor\nvalue = getattr(model, name)\n"
    )
    result = migrate_source(source)
    assert result.source == source.replace("Mass = Quantity | QuantityTensor", "Mass = Quantity")
    assert result.notes == ("1: review retained import bilt.QuantityTensor; cleanup could not safely remove it",)
