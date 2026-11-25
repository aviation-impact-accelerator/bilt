"""Exercise static API checks independently of executing consumer code."""

import ast
import textwrap

import pytest

from bilt._check_api import scan_api


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
        ("from bilt import Quantity as Q; q = Q(1, None); q._value", "BILT102"),
        ("from bilt import PYTHON_BACKEND as backend; backend.transpile", "BILT105"),
        ("obj.__bilt_system_instance_variables__", "BILT103"),
        ("getattr(obj, '__bilt_system_implementation__')", "BILT103"),
        ("from bilt import Implementation; getattr(Implementation, 'evaluate')", "BILT104"),
        ("from bilt import tensor; t = tensor([]); t._values", "BILT102"),
        ("from bilt import Implementation\ndef run(model: Implementation):\n    model.evaluate", "BILT104"),
    ),
)
def test_api_misuse(source: str, code: str) -> None:
    result = scan_api(ast.parse(source), "model.py", "", {})
    assert [item.code for item in result.diagnostics] == [code]
    assert result.diagnostics[0].location.startswith("model.py:")


def test_public_and_unrelated_usage() -> None:
    source = textwrap.dedent("""
        from bilt import Implementation, PYTHON_BACKEND, Quantity
        from bilt.units import meter

        model = Implementation(None, [])
        with model, PYTHON_BACKEND:
            q = Quantity(1, meter)
            q.value
            model.system
        other.evaluate()
        other._value
        def run(model):
            model.evaluate()
        model = other
        model.evaluate()
        """)
    assert not scan_api(ast.parse(source), "model.py", "", {}).diagnostics


def test_imported_model_alias() -> None:
    source = "from .models import motion as model\nmodel.evaluate"
    result = scan_api(ast.parse(source), "app/run.py", "app", {"app.models.motion": "bilt.Implementation"})
    assert [item.code for item in result.diagnostics] == ["BILT104"]
