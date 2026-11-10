"""Top-level model errors should not prevent checking independent later definitions."""

import os
import subprocess
import sys
import textwrap
import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

EXPECTED_ERRORS = 2


def _run(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    (tmp_path / "models.py").write_text(textwrap.dedent(code), encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[tool.bilt.check]\nmodules = ["models"]\n', encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(tmp_path), env.get("PYTHONPATH", ""))))
    return subprocess.run(
        [sys.executable, "-m", "bilt", "check"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_multiple_invalid_equations_in_one_module(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        """
        from typing import Annotated
        from bilt import equation
        from bilt.units import Quantity
        from bilt.units.dimensions import LENGTH, TIME

        Length = Annotated[Quantity, LENGTH]
        Time = Annotated[Quantity, TIME]

        @equation
        def wrong_a(distance: Length, duration: Time) -> bool:
            return distance == duration

        @equation
        def wrong_b(distance: Length, duration: Time) -> bool:
            return duration == distance
        """,
    )
    assert result.returncode == 1
    assert result.stderr.count("BILT001") == EXPECTED_ERRORS
    assert "wrong_a" in result.stderr
    assert "wrong_b" in result.stderr
    assert "2 errors found" in result.stderr
    assert "models.py:11" in result.stderr
    assert "models.py:15" in result.stderr


def test_dependent_implementation_skipped_but_later_equation_checked(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        """
        from typing import Annotated
        from bilt import Implementation, Interface, equation, field, system
        from bilt.units import Quantity
        from bilt.units.dimensions import LENGTH, TIME

        Length = Annotated[Quantity, LENGTH]
        Time = Annotated[Quantity, TIME]

        @system
        class Machine(metaclass=Interface):
            distance: Length = field()
            duration: Time = field()

        @equation
        def broken(distance: Length, duration: Time) -> bool:
            return distance == duration

        implementation = Implementation(Machine, [broken])

        @equation
        def also_broken(distance: Length, duration: Time) -> bool:
            return duration == distance
        """,
    )
    assert result.returncode == 1
    assert result.stderr.count("BILT001") == EXPECTED_ERRORS
    assert "broken" in result.stderr
    assert "also_broken" in result.stderr
    assert "NameError" not in result.stderr
    assert "note: skipped implementation (depends on invalid 'broken')" in result.stderr
    assert "1 definition(s) skipped" in result.stderr


def test_unrelated_name_error_is_not_suppressed(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        """
        from bilt import Interface, field, system

        @system
        class Valid(metaclass=Interface):
            output: float = field()

        x = unknown_variable

        @system
        class AlsoValid(metaclass=Interface):
            output: float = field()
        """,
    )
    assert result.returncode == 1
    assert "NameError" in result.stderr
    assert "unknown_variable" in result.stderr
    assert "Checked 0 module(s)" in result.stdout


def test_regular_import_does_not_suppress_model_validation(tmp_path: Path) -> None:
    (tmp_path / "normal.py").write_text(
        textwrap.dedent("""
        from bilt import equation
        from typing import Annotated
        from bilt.units import Quantity
        from bilt.units.dimensions import LENGTH, TIME
        L = Annotated[Quantity, LENGTH]
        T = Annotated[Quantity, TIME]

        @equation
        def wrong(a: L, b: T) -> bool:
            return a == b
        """),
        encoding="utf-8",
    )
    (tmp_path / "models.py").write_text("import normal\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[tool.bilt.check]\nmodules = ["models"]\n', encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(tmp_path), env.get("PYTHONPATH", ""))))
    result = subprocess.run(
        [sys.executable, "-m", "bilt", "check"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "normal.py:" in result.stderr
    assert "BILT001" in result.stderr


def test_unrelated_bilt_error_is_not_swallowed(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        """
        from bilt import equation
        from bilt.core._errors import BiltDefinitionError

        raise BiltDefinitionError("unrelated module setup failure")

        @equation
        def later(a: float) -> bool:
            return a == 1
        """,
    )
    assert result.returncode == 1
    assert "unrelated module setup failure" in result.stderr
    assert "Checked 0 module(s)" in result.stdout


def test_failed_definition_does_not_repeat_import_side_effects(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        """
        from pathlib import Path
        from typing import Annotated
        from bilt import equation
        from bilt.units import Quantity
        from bilt.units.dimensions import LENGTH, TIME

        count = Path(__file__).with_suffix(".count")
        count.write_text((count.read_text() if count.exists() else "") + "x")
        Length = Annotated[Quantity, LENGTH]
        Time = Annotated[Quantity, TIME]

        @equation
        def wrong(a: Length, b: Time) -> bool:
            return a == b

        @equation
        def valid(a: Length) -> bool:
            return a == 0
        """,
    )
    assert result.returncode == 1
    assert "BILT001" in result.stderr
    assert (tmp_path / "models.count").read_text() == "x"
    assert "Checked 1 module(s)" in result.stdout
    assert "1 equation(s)" in result.stdout


def test_package_resources_work_with_recovery(tmp_path: Path) -> None:
    package = tmp_path / "models"
    package.mkdir()
    (package / "data.txt").write_text("hello from resource", encoding="utf-8")
    (package / "__init__.py").write_text(
        textwrap.dedent("""
        from importlib.resources import files
        from typing import Annotated
        from bilt import equation
        from bilt.units import Quantity
        from bilt.units.dimensions import LENGTH, TIME

        assert files(__package__).joinpath("data.txt").read_text() == "hello from resource"

        Length = Annotated[Quantity, LENGTH]
        Time = Annotated[Quantity, TIME]

        @equation
        def wrong(a: Length, b: Time) -> bool:
            return a == b
        """),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text('[tool.bilt.check]\nmodules = ["models"]\n', encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(tmp_path), env.get("PYTHONPATH", ""))))
    result = subprocess.run(
        [sys.executable, "-m", "bilt", "check"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "BILT001" in result.stderr
    assert "AssertionError" not in result.stderr


def test_qualified_bilt_decorator_is_recovered(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        """
        import bilt as model
        from typing import Annotated
        from bilt.units import Quantity
        from bilt.units.dimensions import LENGTH, TIME

        Length = Annotated[Quantity, LENGTH]
        Time = Annotated[Quantity, TIME]

        @model.equation
        def wrong_a(a: Length, b: Time) -> bool:
            return a == b

        @model.equation
        def wrong_b(a: Length, b: Time) -> bool:
            return b == a
        """,
    )
    assert result.returncode == 1
    assert result.stderr.count("BILT001") == EXPECTED_ERRORS


def test_ordinary_python_usage_of_failed_definition_is_not_suppressed(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        """
        from bilt import equation
        from typing import Annotated
        from bilt.units import Quantity
        from bilt.units.dimensions import LENGTH, TIME

        Length = Annotated[Quantity, LENGTH]
        Time = Annotated[Quantity, TIME]

        @equation
        def broken(a: Length, b: Time) -> bool:
            return a == b

        reference = broken

        @equation
        def later(a: Length, b: Time) -> bool:
            return b == a
        """,
    )
    assert result.returncode == 1
    assert "BILT001" in result.stderr
    assert "NameError" in result.stderr
    assert "definition(s) skipped" not in result.stderr


def test_invalid_system_ports_are_reported_together(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        """
        from typing import Annotated
        from bilt import Interface, field, system
        from bilt.units import Quantity
        from bilt.units.dimensions import LENGTH

        @system
        class Broken(metaclass=Interface):
            first: Quantity = field()
            second: Annotated[float, LENGTH] = field()
        """,
    )
    assert result.returncode == 1
    assert result.stderr.count("BILT003") == EXPECTED_ERRORS
    assert "Port 'first'" in result.stderr
    assert "Port 'second'" in result.stderr
    assert "2 errors found" in result.stderr
