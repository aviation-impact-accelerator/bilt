"""Confirm the foundation's dependencies rather than just its import order."""

import ast
import subprocess
import sys
from pathlib import Path

ARCH = Path(__file__).resolve().parents[1] / "architecture"
REPO = ARCH.parents[2]


def test_tensor_foundation_has_no_unit_or_quantity_imports() -> None:
    """Catch direct and eager transitive imports in the foundational module."""
    tree = ast.parse((ARCH / "_tensor.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not (
                node.module.startswith("bilt.units") or "quantity" in node.module
            )
        if isinstance(node, ast.Import):
            assert all(not (alias.name.startswith("bilt.units") or "quantity" in alias.name) for alias in node.names)


def test_fresh_process_tensor_import_does_not_initialise_units() -> None:
    """Ensure importing the tensor foundation doesn't eagerly load units."""
    script = (
        "import sys; "
        "import prototypes.quantity_abc.architecture._tensor; "
        "assert 'bilt.units' not in sys.modules; "
        "assert 'prototypes.quantity_abc.architecture._quantity' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_imports_in_both_orders() -> None:
    """Exercise explicit, clean interpreter imports in both directions."""
    for imports in (
        "from prototypes.quantity_abc.architecture import _tensor, _quantity, _quantity_tensor",
        "from prototypes.quantity_abc.architecture import _quantity_tensor, _quantity, _tensor",
    ):
        result = subprocess.run(
            [sys.executable, "-c", imports],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
