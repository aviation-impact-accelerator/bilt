import subprocess
import sys
import textwrap

import pytest

import bilt
from bilt._backend._jax import JAX_BACKEND

_EXTRA_PACKAGES = {"jax": ("jax", "jaxlib")}


def _run_without(extra: str, code: str) -> subprocess.CompletedProcess[str]:
    blocked = "".join(f"sys.modules[{package!r}] = None\n" for package in _EXTRA_PACKAGES[extra])
    script = "import sys\n" + blocked + textwrap.dedent(code)
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # noqa: S603


@pytest.mark.parametrize("extra", sorted({extra for _, extra in bilt._OPTIONAL.values()}))  # noqa: SLF001
def test_import_bilt_and_evaluate_on_python_backend_without_extra(extra: str) -> None:
    result = _run_without(
        extra,
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


@pytest.mark.parametrize(("name", "extra"), [(name, extra) for name, (_, extra) in bilt._OPTIONAL.items()])  # noqa: SLF001
def test_optional_name_without_extra_raises_import_error_naming_extra(name: str, extra: str) -> None:
    result = _run_without(
        extra,
        f"""
        import bilt

        try:
            getattr(bilt, {name!r})
        except ImportError as error:
            assert "bilt[{extra}]" in str(error), error
            assert isinstance(error.__cause__, ModuleNotFoundError)
        else:
            raise AssertionError("expected ImportError")
        """,
    )
    assert result.returncode == 0, result.stderr


def test_optional_names_are_public() -> None:
    assert set(bilt._OPTIONAL) <= set(bilt.__all__)  # noqa: SLF001


def test_jax_backend_is_adapter_instance_when_jax_installed() -> None:
    assert bilt.JAX_BACKEND is JAX_BACKEND


def test_missing_bilt_module_is_not_reported_as_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(bilt._OPTIONAL, "BROKEN", ("bilt._no_such_module", "broken"))  # noqa: SLF001
    with pytest.raises(ModuleNotFoundError, match=r"bilt\._no_such_module"):
        _ = bilt.BROKEN


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no_such_name"):
        _ = bilt.no_such_name
