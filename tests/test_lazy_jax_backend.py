import subprocess
import sys
import textwrap

import pytest

import bilt
from bilt._backend._jax import JAX_BACKEND


def _run_without_jax(code: str) -> subprocess.CompletedProcess[str]:
    script = 'import sys\nsys.modules["jax"] = None\n' + textwrap.dedent(code)
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # noqa: S603


def test_import_bilt_and_evaluate_on_python_backend_without_jax() -> None:
    result = _run_without_jax(
        """
        from bilt import PYTHON_BACKEND, Implementation, Interface, field, system, transform

        @system
        class Doubler(metaclass=Interface):
            x: float = field()
            y: float = field()

        @transform
        def y(x: float) -> float:
            return 2.0 * x

        with PYTHON_BACKEND, Implementation(Doubler, [y]):
            assert Doubler(x=3.0).y == 6.0
        """,
    )
    assert result.returncode == 0, result.stderr


def test_jax_backend_without_jax_raises_import_error_naming_extra() -> None:
    result = _run_without_jax(
        """
        import bilt

        try:
            bilt.JAX_BACKEND
        except ImportError as error:
            assert "bilt[jax]" in str(error), error
            assert isinstance(error.__cause__, ModuleNotFoundError)
        else:
            raise AssertionError("expected ImportError")
        """,
    )
    assert result.returncode == 0, result.stderr


def test_jax_backend_is_adapter_instance_when_jax_installed() -> None:
    assert bilt.JAX_BACKEND is JAX_BACKEND


def test_missing_bilt_module_is_not_reported_as_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(bilt._OPTIONAL, "BROKEN", ("bilt._no_such_module", "broken"))  # noqa: SLF001
    with pytest.raises(ModuleNotFoundError, match=r"bilt\._no_such_module"):
        _ = bilt.BROKEN


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no_such_name"):
        _ = bilt.no_such_name
