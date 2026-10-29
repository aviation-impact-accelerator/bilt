"""Execute the code shown in the linear guide, coupled springs and recuperated turbine examples."""

import re
import sys
import types
import typing
from pathlib import Path

import pytest

from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from bilt._backend import Backend

DOCS = Path(__file__).resolve().parents[2] / "docs"


def _cells(text: str) -> Iterator[str]:
    for match in re.finditer(r"```python exec[^\n]*\n(.*?)\n```", text, re.DOTALL):
        yield match.group(1)


def _run(cells: Iterator[str], module: types.ModuleType, backend: Backend) -> None:
    for index, cell in enumerate(cells):
        # Imports in the documentation name PYTHON_BACKEND. Substitute only the
        # selected backend so the same published model/code runs on both engines.
        exec(compile(cell, f"{module.__name__}:cell{index}", "exec"), module.__dict__)  # noqa: S102 - trusted repository examples
        module.__dict__["PYTHON_BACKEND"] = backend


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_linear_guide_examples(backend: Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("_bilt_linear_guide")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    _run(_cells((DOCS / "guide/coupled-linear-systems.md").read_text()), module, backend)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_recuperated_turbine_example(backend: Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    text = (DOCS / "worked_examples/two-shaft-gas-turbine.md").read_text()
    source = re.search(r'model_source = """(.*?)"""', text, re.DOTALL)
    assert source is not None
    model_module = types.ModuleType("_bilt_turbine_model")
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    exec(compile(source.group(1), "gas_turbine_model", "exec"), model_module.__dict__)  # noqa: S102 - trusted repository model
    module = types.ModuleType("_bilt_recuperated_example")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__["model_module"] = model_module
    setup = """
import numpy as np
from bilt import Quantity
from bilt.units import UNITLESS, absolute_kelvin, kelvin, kilogram, kilojoule, megawatt, second
"""
    exec(setup, module.__dict__)  # noqa: S102 - trusted example imports
    cells = _cells(text[text.index("## 14.") :])
    # The example's first cell imports PYTHON_BACKEND and immediately evaluates.
    # Rebind that import for the JAX run without changing the published model.
    first = next(cells).replace(
        "from bilt import Implementation, Interface, PYTHON_BACKEND, equation, field, system",
        "from bilt import Implementation, Interface, equation, field, system",
    )
    module.__dict__["PYTHON_BACKEND"] = backend
    _run(iter((first, *cells)), module, backend)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_coupled_springs_example(backend: Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    text = (DOCS / "worked_examples/two-springs-in-series.md").read_text()
    source = re.search(r'model_source = """(.*?)"""', text, re.DOTALL)
    assert source is not None
    model_module = types.ModuleType("_bilt_springs_model")
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    exec(compile(source.group(1), "springs_model", "exec"), model_module.__dict__)  # noqa: S102 - trusted repository model
    module = types.ModuleType("_bilt_coupled_springs")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(model_module.__dict__)
    setup = "import numpy as np\nfrom bilt.units import meter, newton"
    exec(setup, module.__dict__)  # noqa: S102 - trusted example imports
    section = text[text.index("## 9.") : text.index("## 10.")]
    with backend:
        _run(_cells(section), module, backend)
