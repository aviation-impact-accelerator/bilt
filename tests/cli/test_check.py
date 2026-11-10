"""Exercise bilt check against small consumer model projects."""

import os
import subprocess
import sys
import textwrap
import typing

import pytest

from bilt._cli import _settings
from bilt.errors import BiltCheckConfigurationError

if typing.TYPE_CHECKING:
    from pathlib import Path

CONFIG_ERROR = 2


def _invoke(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), env.get("PYTHONPATH", ""))))
    return subprocess.run(  # noqa: S603 - fixed interpreter and module; arguments controlled by tests
        [sys.executable, "-m", "bilt", "check", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _run_check(
    tmp_path: Path, *, source: str, module: str = "sample", config: str | None = None
) -> subprocess.CompletedProcess[str]:
    package = tmp_path / module
    package.mkdir()
    (package / "__init__.py").write_text(textwrap.dedent(source), encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        config if config is not None else f'[tool.bilt.check]\nmodules = ["{module}"]\n',
        encoding="utf-8",
    )
    return _invoke(tmp_path)


def test_valid_model_passes(tmp_path: Path) -> None:
    result = _run_check(
        tmp_path,
        source="""
        from bilt import Interface, Implementation, equation, field, system, transform

        @system
        class Travel(metaclass=Interface):
            distance: float = field()
            speed: float = field()
            time: float = field()

        @equation
        def motion(distance: float, speed: float, time: float) -> bool:
            return distance == speed * time

        @transform
        def speed(distance: float, time: float) -> float:
            return distance / time

        basic = Implementation(Travel, [motion, speed])
    """,
    )
    assert result.returncode == 0, result.stderr
    assert "1 system(s), 1 implementation(s), 1 equation(s), 1 transform(s)" in result.stdout
    assert "No definition errors found" in result.stdout


def test_equation_dimensionality_error_has_source_location(tmp_path: Path) -> None:
    result = _run_check(
        tmp_path,
        source="""
        from typing import Annotated
        from bilt import equation
        from bilt.units import Quantity
        from bilt.units.dimensions import LENGTH, TIME

        Length = Annotated[Quantity, LENGTH]
        Time = Annotated[Quantity, TIME]

        @equation
        def wrong(length: Length, duration: Time) -> bool:
            return length == duration
    """,
    )
    assert result.returncode == 1
    assert "sample/__init__.py:" in result.stderr
    assert "BiltEquationDimensionalityError" in result.stderr


def test_transform_definition_error_is_reported(tmp_path: Path) -> None:
    result = _run_check(
        tmp_path,
        source="""
        from typing import Annotated
        from bilt import transform
        from bilt.units import Quantity
        from bilt.units.dimensions import LENGTH

        Length = Annotated[Quantity, LENGTH]

        @transform
        def distance(length: Length) -> Length:
            return length.value
    """,
    )
    assert result.returncode == 1
    assert "BiltDefinitionError" in result.stderr


def test_incompatible_implementation_declarations_are_reported(tmp_path: Path) -> None:
    result = _run_check(
        tmp_path,
        source="""
        from typing import Annotated
        from bilt import Implementation, Interface, field, system, transform
        from bilt.units import Quantity
        from bilt.units.dimensions import LENGTH, TIME

        Length = Annotated[Quantity, LENGTH]
        Time = Annotated[Quantity, TIME]

        @system
        class Model(metaclass=Interface):
            distance: Length = field()
            elapsed: Time = field()

        @transform
        def distance(elapsed: Time) -> Time:
            return elapsed

        basic = Implementation(Model, [distance])
    """,
    )
    assert result.returncode == 1
    assert "BiltImplementationDimensionalityError" in result.stderr


def test_recursive_discovery_imports_submodules(tmp_path: Path) -> None:
    package = tmp_path / "models"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "one.py").write_text(
        textwrap.dedent("""
        from bilt import Interface, field, system

        @system
        class Model(metaclass=Interface):
            output: float = field()
    """),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text('[tool.bilt.check]\nmodules = ["models"]\n', encoding="utf-8")
    result = _invoke(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Checked 2 module(s): 1 system(s)" in result.stdout


def test_independent_module_failures_are_collected(tmp_path: Path) -> None:
    (tmp_path / "bad_a.py").write_text("raise RuntimeError('first')\n", encoding="utf-8")
    (tmp_path / "bad_b.py").write_text("raise RuntimeError('second')\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[tool.bilt.check]\nmodules = ["bad_a", "bad_b"]\n', encoding="utf-8")
    result = _invoke(tmp_path)
    assert result.returncode == 1
    assert "RuntimeError: first" in result.stderr
    assert "RuntimeError: second" in result.stderr


def test_no_models_is_a_failure(tmp_path: Path) -> None:
    result = _run_check(tmp_path, source="VALUE = 1")
    assert result.returncode == 1
    assert "no Bilt model definitions" in result.stderr


def test_check_does_not_require_model_configuration(tmp_path: Path) -> None:
    result = _run_check(
        tmp_path,
        config="[project]\nname = 'example'\n",
        source="""
        from bilt import Interface, field, system

        @system
        class Model(metaclass=Interface):
            output: float = field()
        """,
    )
    assert result.returncode == 0, result.stderr
    assert "1 system(s)" in result.stdout


def test_missing_module_is_an_import_error(tmp_path: Path) -> None:
    result = _run_check(tmp_path, source="VALUE = 1", config='[tool.bilt.check]\nmodules = ["missing"]\n')
    assert result.returncode == 1
    assert "ModuleNotFoundError" in result.stderr


def test_cli_module_override(tmp_path: Path) -> None:
    (tmp_path / "chosen.py").write_text(
        textwrap.dedent("""
        from bilt import Interface, field, system

        @system
        class Model(metaclass=Interface):
            output: float = field()
    """),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text('[tool.bilt.check]\nmodules = ["missing"]\n', encoding="utf-8")
    result = _invoke(tmp_path, "--module", "chosen")
    assert result.returncode == 0, result.stderr


def test_invalid_configuration_is_a_usage_error(tmp_path: Path) -> None:
    result = _run_check(tmp_path, source="VALUE = 1", config="[tool.bilt.check]\nmodules = 3\n")
    assert result.returncode == CONFIG_ERROR
    assert "'modules'" in result.stderr


def test_invalid_settings_raise_bilt_configuration_error() -> None:
    with pytest.raises(BiltCheckConfigurationError, match="modules"):
        _settings({"tool": {"bilt": {"check": {"modules": 3}}}}, None)


def test_check_without_pyproject_discovers_models(tmp_path: Path) -> None:
    (tmp_path / "model.py").write_text(
        "from bilt import Interface, field, system\n"
        "@system\n"
        "class Model(metaclass=Interface):\n"
        "    output: float = field()\n",
        encoding="utf-8",
    )
    result = _invoke(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "1 system(s)" in result.stdout


def test_auto_discovery_finds_new_model_without_updating_config(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.bilt.check]\n", encoding="utf-8")
    for name in ("first", "second"):
        (tmp_path / f"{name}.py").write_text(
            "from bilt import Interface, field, system\n"
            "@system\n"
            f"class {name.title()}(metaclass=Interface):\n"
            "    output: float = field()\n",
            encoding="utf-8",
        )
    result = _invoke(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "2 system(s)" in result.stdout


def test_discovery_ignores_unrelated_scripts_and_tests(tmp_path: Path) -> None:
    (tmp_path / "models.py").write_text(
        "from bilt import Interface, field, system\n"
        "@system\n"
        "class Model(metaclass=Interface):\n"
        "    output: float = field()\n",
        encoding="utf-8",
    )
    (tmp_path / "unrelated.py").write_text("raise RuntimeError('unrelated import')\n", encoding="utf-8")
    testdir = tmp_path / "tests"
    testdir.mkdir()
    (testdir / "test_broken.py").write_text("raise RuntimeError('test import')\n", encoding="utf-8")
    result = _invoke(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "RuntimeError" not in result.stderr
    assert "1 system(s)" in result.stdout


def test_discovery_supports_src_layout(tmp_path: Path) -> None:
    package = tmp_path / "src" / "example"
    package.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "model.py").write_text(
        "from bilt import Interface, field, system\n"
        "@system\n"
        "class Model(metaclass=Interface):\n"
        "    output: float = field()\n",
        encoding="utf-8",
    )
    result = _invoke(tmp_path, "src")
    assert result.returncode == 0, result.stderr
    assert "1 system(s)" in result.stdout


def test_discovery_accepts_subdirectory_and_excludes(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name in ("good", "excluded"):
        (package / f"{name}.py").write_text(
            "from bilt import Interface, field, system\n"
            "@system\n"
            f"class {name.title()}(metaclass=Interface):\n"
            "    output: float = field()\n",
            encoding="utf-8",
        )
    (tmp_path / "pyproject.toml").write_text('[tool.bilt.check]\nexclude = ["package/excluded.py"]\n', encoding="utf-8")
    result = _invoke(tmp_path, "package")
    assert result.returncode == 0, result.stderr
    assert "1 system(s)" in result.stdout


def test_invalid_source_is_reported_instead_of_skipped(tmp_path: Path) -> None:
    (tmp_path / "model.py").write_text("def invalid(:\n", encoding="utf-8")
    result = _invoke(tmp_path)
    assert result.returncode == 1
    assert "model.py:1: error: SyntaxError" in result.stderr


def test_explicit_module_extends_auto_discovery(tmp_path: Path) -> None:
    (tmp_path / "direct.py").write_text(
        "from bilt import Interface, field, system\n"
        "@system\n"
        "class Direct(metaclass=Interface):\n"
        "    output: float = field()\n",
        encoding="utf-8",
    )
    (tmp_path / "generated.py").write_text(
        "from bilt import Interface, field, system\n"
        "def make_model():\n"
        "    @system\n"
        "    class Generated(metaclass=Interface):\n"
        "        output: float = field()\n"
        "    return Generated\n"
        "constructed = make_model()\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text('[tool.bilt.check]\nmodules = ["generated"]\n', encoding="utf-8")
    result = _invoke(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "2 system(s)" in result.stdout
