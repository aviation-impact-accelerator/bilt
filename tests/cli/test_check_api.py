"""Exercise static API checks independently of executing consumer code."""

import ast
import textwrap
from typing import TYPE_CHECKING

import pytest

from bilt._check_api import scan_api
from bilt._check_discovery import discover
from bilt._migrate import migrate_source

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("source", "code"),
    (
        ("from bilt.core._system import Implementation", "BILT101"),
        ("import bilt._backend as backend", "BILT101"),
        ("from bilt import core", "BILT101"),
        ("import bilt as b; b.core", "BILT101"),
        ("from bilt import Implementation as I; I.evaluate", "BILT104"),
        ("from bilt import Implementation as I; model = I(None, []); model.evaluate", "BILT104"),
        ("from bilt import Implementation; model = Implementation(None, []); model._plan_cache", "BILT102"),
        ("from bilt import Quantity as Q; q: Q; q._value", "BILT102"),
        ("from bilt import PYTHON_BACKEND as backend; backend.transpile", "BILT105"),
        ("obj.__bilt_system_instance_variables__", "BILT103"),
        ("getattr(obj, '__bilt_system_implementation__')", "BILT103"),
        ("from bilt import Implementation; getattr(Implementation, 'evaluate')", "BILT104"),
        ("from bilt import tensor; t = tensor([]); t._values", "BILT102"),
        ("from bilt import Implementation\ndef run(model: Implementation):\n    model.evaluate", "BILT104"),
    ),
)
def test_api_misuse(source: str, code: str) -> None:
    result = scan_api(source, ast.parse(source), "model.py", "", {})
    assert [item.code for item in result.diagnostics] == [code]
    assert result.diagnostics[0].location.startswith("model.py:")


def test_public_and_unrelated_usage() -> None:
    source = textwrap.dedent("""
        from bilt import Implementation, PYTHON_BACKEND, Quantity
        from bilt.units import meter

        model = Implementation(None, [])
        with model, PYTHON_BACKEND:
            q: Quantity = 1.0 * meter
            q.value
            model.system
        other.evaluate()
        other._value
        def run(model):
            model.evaluate()
        model = other
        model.evaluate()
        """)
    assert not scan_api(source, ast.parse(source), "model.py", "", {}).diagnostics


def test_imported_model_alias() -> None:
    source = "from .models import motion as model\nmodel.evaluate"
    result = scan_api(source, ast.parse(source), "app/run.py", "app", {"app.models.motion": "bilt.Implementation"})
    assert [item.code for item in result.diagnostics] == ["BILT104"]


@pytest.mark.parametrize(
    "source",
    (
        "from bilt import Quantity; Quantity(1, None)",
        "from bilt.units import Quantity as Q; Q(1, None)",
        "import bilt as b; b.Quantity(1, None)",
        "from bilt import Tensor as T; T.scalar(2.0)",
        "from bilt import QuantityTensor; QuantityTensor()",
        "from bilt import Number; Number(2)",
        "from bilt import Quantity; Q = Quantity; Q(1, None)",
        "from bilt import Tensor; scalar = Tensor.scalar; scalar(1)",
    ),
)
def test_obsolete_protocol_constructors(source: str) -> None:
    result = scan_api(source, ast.parse(source), "model.py", "", {})
    assert [item.code for item in result.diagnostics] == ["BILT106"]
    assert "instead" in result.diagnostics[0].message


@pytest.mark.parametrize(
    "source",
    (
        "from bilt import Quantity as Q; Q = other; Q(1)",
        "from bilt import Quantity as Q\ndef run(Q):\n    Q(1)",
        "from bilt import Quantity as Q\ndef run(value: Q):\n    value(1)",
        "from bilt import Tensor, tensor; t = tensor(1); t.scalar(2)",
        "from bilt import Quantity, QuantityTensor; Old = Quantity | QuantityTensor",
        "from bilt import tensor; tensor(1)",
        "from bilt import Quantity as Q\nfactory = lambda Q: Q(2, meter)",
        "from bilt import Quantity as Q\nvalues = [Q(2, meter) for Q in factories]",
    ),
)
def test_migration_checks_do_not_flag_unrelated_calls(source: str) -> None:
    assert not scan_api(source, ast.parse(source), "model.py", "", {}).diagnostics


@pytest.mark.parametrize(
    ("source", "manual"),
    (
        ("from bilt import Quantity\nx = Quantity(2.0, meter)\n", False),
        ("from bilt.units import Quantity as Q\nx = Q(2.0, meter)\n", False),
        ("import bilt as b\nx = b.Tensor.scalar(2)\n", False),
        ("from bilt import Tensor\nx = Tensor(values, domain)\n", False),
        ("from bilt import QuantityTensor\nx = QuantityTensor(values, domain, meter)\n", False),
        ("from bilt import Number\nx = Number(2)\n", True),
        ("from bilt import Quantity\nx = Quantity(unit=u(), value=v())\n", True),
        ("from bilt import QuantityTensor\nx = QuantityTensor(values, domain)\n", True),
        ("from bilt import Quantity as Q\ndef run():\n    return Q(2, meter)\n", False),
        ("from bilt import Quantity as Q\nx = [Q(v, meter) for v in values]\n", False),
    ),
)
def test_obsolete_constructor_check_and_migration_agree(source: str, manual: bool) -> None:
    before = scan_api(source, ast.parse(source), "model.py", "", {})
    assert [item.code for item in before.diagnostics] == ["BILT106"]
    migrated = migrate_source(source)
    after = scan_api(migrated.source, ast.parse(migrated.source), "model.py", "", {})
    remaining = [item for item in after.diagnostics if item.code == "BILT106"]
    if manual:
        assert remaining
        assert any("review obsolete" in note for note in migrated.notes)
    else:
        assert not remaining
        assert not migrated.notes
        assert migrate_source(migrated.source) == migrated


def test_discovery_reports_invalid_lexical_scope(tmp_path: Path) -> None:
    (tmp_path / "invalid.py").write_text("def run():\n    nonlocal missing\n", encoding="utf-8")
    result = discover(tmp_path, tmp_path, ())
    assert len(result.errors) == 1
    assert result.errors[0].startswith("invalid.py:2: error: SyntaxError:")
