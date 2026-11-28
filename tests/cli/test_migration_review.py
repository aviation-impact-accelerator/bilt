"""Review diagnostics are advisory and do not invent scalar semantics."""

from bilt._migrate import migrate_source


def test_quantity_runtime_checks_alone_do_not_require_review() -> None:
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
    assert not result.notes
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


def test_scalar_return_after_quantity_check_is_reported() -> None:
    source = (
        "from bilt import Quantity\n"
        "def extract(value) -> float:\n"
        "    assert isinstance(value, Quantity)\n"
        "    return value.convert_to(percent).value\n"
    )
    result = migrate_source(source)
    assert result.source == source
    assert result.notes == (
        "4: review magnitude at scalar or NumPy boundary; explicitly narrow or extract the intended shape",
    )
    corrected = source.replace(
        "    return value.convert_to(percent).value",
        "    magnitude = value.convert_to(percent).value\n    assert isinstance(magnitude, float)\n    return magnitude",
    )
    assert not migrate_source(corrected).notes


def test_numpy_boundary_after_negative_tensor_check() -> None:
    source = (
        "import numpy as np\n"
        "from bilt import QuantityTensor\n"
        "def extract(converted):\n"
        "    if not isinstance(converted, QuantityTensor):\n"
        "        return np.broadcast_to(converted.value, shape)\n"
        "    return converted.values\n"
    )
    result = migrate_source(source)
    assert result.source == source
    assert result.notes == (
        "5: review magnitude at scalar or NumPy boundary; explicitly narrow or extract the intended shape",
    )
    corrected = source.replace(
        "        return np.broadcast_to(converted.value, shape)",
        "        magnitude = converted.value\n"
        "        assert isinstance(magnitude, float)\n"
        "        return np.broadcast_to(magnitude, shape)",
    )
    assert not migrate_source(corrected).notes


def test_numpy_boundary_after_quantity_assertion() -> None:
    source = (
        "import numpy as np\n"
        "from bilt import Quantity\n"
        "def check(efficiency):\n"
        "    assert isinstance(efficiency, Quantity)\n"
        "    np.testing.assert_allclose(efficiency.convert_to(percent).value, 2.0)\n"
    )
    assert migrate_source(source).notes == (
        "5: review magnitude at scalar or NumPy boundary; explicitly narrow or extract the intended shape",
    )


def test_no_review_for_unrelated_magnitude_or_shape_independent_use() -> None:
    source = (
        "import numpy as np\n"
        "from bilt import Quantity\n"
        "def unrelated(value) -> float:\n"
        "    return value.value\n"
        "def check(value):\n"
        "    assert isinstance(value, Quantity)\n"
        "    assert value.convert_to(percent).value == expected\n"
        "    return value.value\n"
        "def explicit(value: Quantity) -> float:\n"
        "    return float(value.value)\n"
        "def unrelated_numpy(value):\n"
        "    return np.broadcast_to(value.value, shape)\n"
    )
    assert not migrate_source(source).notes
