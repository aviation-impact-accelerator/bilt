"""Source-only migration regression tests."""

import typing

from bilt._cli import main
from bilt._migrate import migrate_source

if typing.TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_quantity_constructor_and_union() -> None:
    source = "from bilt import Quantity as Q, QuantityTensor as QT\nMass = Q | QT\nx = Q(2.0, meter)\n"
    result = migrate_source(source)
    assert "Mass = Q\n" in result.source
    assert "x = ((2.0) * (meter))" in result.source
    assert not result.notes
    assert migrate_source(result.source).source == result.source


def test_module_alias_tensor_scalar() -> None:
    source = "import bilt as b\nx = b.Tensor.scalar(2, unit=meter)\n"
    assert "x = b.tensor(2, unit=meter)" in migrate_source(source).source


def test_unrelated_shadowed_constructor() -> None:
    source = "from bilt import Quantity as Q\nQ = other\nx = Q(2, meter)\n"
    assert migrate_source(source).source == source


def test_manual_review_preserves_comments() -> None:
    source = "from bilt import Quantity\nx = Quantity(\n    2,  # keep this\n    meter,\n)\n"
    result = migrate_source(source)
    assert result.source == source
    assert result.notes


def test_utf8_offsets_and_line_endings() -> None:
    source = "from bilt import Quantity\r\nname = 'é'; x = Quantity(2, meter)\r\n"
    assert migrate_source(source).source == "from bilt import Quantity\r\nname = 'é'; x = ((2) * (meter))\r\n"


def test_cli_preview_and_write(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "model.py"
    original = "from bilt import Quantity\nx = Quantity(2, meter)\n"
    path.write_text(original, encoding="utf-8")
    assert main(["migrate", str(path)]) == 0
    assert path.read_text(encoding="utf-8") == original
    assert "Would update 1 file(s)" in capsys.readouterr().out
    assert main(["migrate", str(path), "--write"]) == 0
    assert "x = ((2) * (meter))" in path.read_text(encoding="utf-8")
    assert main(["migrate", str(path), "--write"]) == 0
    assert "Updated 0 file(s)" in capsys.readouterr().out


def test_keyword_quantity_and_evaluation_order() -> None:
    source = "from bilt import Quantity\nx = Quantity(value=2, unit=meter)\ny = Quantity(unit=u(), value=v())\n"
    result = migrate_source(source)
    assert "x = ((2) * (meter))" in result.source
    assert "y = Quantity(unit=u(), value=v())" in result.source
    assert len(result.notes) == 1


def test_direct_tensor_constructors_insert_import() -> None:
    source = "from bilt import Tensor, QuantityTensor\nx = Tensor(v, d)\ny = QuantityTensor(v, d, u)\n"
    result = migrate_source(source)
    assert result.source.startswith("from bilt import tensor\n")
    assert "x = tensor(values=v, domain=d)" in result.source
    assert "y = tensor(values=v, domain=d, unit=u)" in result.source
    assert not result.notes
    assert migrate_source(result.source).source == result.source


def test_tensor_factory_import_collision_and_future() -> None:
    source = '"""Docstring."""\nfrom __future__ import annotations\nfrom bilt import Tensor\ntensor = other\nx = Tensor.scalar(2, u)\n'
    result = migrate_source(source)
    assert "from __future__ import annotations\nfrom bilt import tensor as _tensor\n" in result.source
    assert "x = _tensor(2, unit=u)" in result.source
    assert not result.notes


def test_dimensioned_tensor_annotation() -> None:
    source = (
        "from typing import Annotated\nfrom bilt import Tensor\n"
        "from bilt.units.dimensions import MASS\n"
        "Mass = Annotated[Tensor, MASS]\nPlain = Annotated[Tensor, 'metadata']\n"
    )
    result = migrate_source(source)
    assert "from bilt import QuantityTensor\n" in result.source
    assert "Mass = Annotated[QuantityTensor, MASS]" in result.source
    assert "Plain = Annotated[Tensor, 'metadata']" in result.source
    assert migrate_source(result.source).source == result.source


def test_multiline_union() -> None:
    source = "from bilt import Quantity, QuantityTensor\nMass = (\n    Quantity |\n    QuantityTensor\n)\n"
    assert "Mass = (\n    Quantity\n)" in migrate_source(source).source


def test_migrate_functions_with_local_assignments() -> None:
    source = (
        "from bilt import Quantity\n"
        "def model(x):\n"
        "    value = x + 1\n"
        "    return Quantity(value, meter)\n"
    )
    result = migrate_source(source)
    assert "return ((value) * (meter))" in result.source
    assert not result.notes
    assert migrate_source(result.source).source == result.source


def test_lexically_shadowed_constructor_is_not_rewritten() -> None:
    source = (
        "from bilt import Quantity as Q\n"
        "def model():\n"
        "    x = Q(2, meter)\n"
        "    Q = other\n"
        "    return x\n"
        "def by_argument(Q):\n"
        "    return Q(2, meter)\n"
        "def by_branch(flag):\n"
        "    if flag:\n"
        "        Q = other\n"
        "    return Q(2, meter)\n"
    )
    assert migrate_source(source).source == source


def test_global_reassignment_is_not_rewritten() -> None:
    source = "from bilt import Quantity as Q\ndef model():\n    return Q(2, meter)\nQ = other\n"
    assert migrate_source(source).source == source


def test_nested_function_free_variable_is_not_rewritten() -> None:
    source = (
        "from bilt import Quantity as Q\n"
        "def outer(Q):\n"
        "    def inner():\n"
        "        return Q(2, meter)\n"
        "    return inner\n"
    )
    assert migrate_source(source).source == source


def test_method_does_not_inherit_class_constructor_binding() -> None:
    source = (
        "from bilt import Quantity as Q\n"
        "class Model:\n"
        "    Q = other\n"
        "    def run(self):\n"
        "        x = 2\n"
        "        return Q(x, meter)\n"
    )
    assert "return ((x) * (meter))" in migrate_source(source).source


def test_unconditional_function_local_import() -> None:
    source = "def model():\n    from bilt import Quantity as Q\n    return Q(2, meter)\n"
    result = migrate_source(source)
    assert "return ((2) * (meter))" in result.source
    assert not result.notes


def test_conditional_local_import_is_not_rewritten() -> None:
    source = "def model(flag):\n    if flag:\n        from bilt import Quantity as Q\n    return Q(2, meter)\n"
    assert migrate_source(source).source == source


def test_async_function_with_locals() -> None:
    source = "from bilt import Quantity\nasync def model():\n    x = 2\n    return Quantity(x, meter)\n"
    assert "return ((x) * (meter))" in migrate_source(source).source


def test_explicit_global_write_is_not_rewritten() -> None:
    source = (
        "from bilt import Quantity as Q\n"
        "def change():\n"
        "    global Q\n"
        "    Q = other\n"
        "def model():\n"
        "    return Q(2, meter)\n"
    )
    assert migrate_source(source).source == source


def test_defaults_are_evaluated_in_enclosing_scope() -> None:
    source = "from bilt import Quantity as Q\ndef model(Q=Q(2, meter)):\n    return Q(2, meter)\n"
    result = migrate_source(source)
    assert "def model(Q=((2) * (meter))):" in result.source
    assert "return Q(2, meter)" in result.source
