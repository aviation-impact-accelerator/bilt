"""The CLI prints independent validation errors from a single definition."""

import os
import subprocess
import sys
import textwrap
import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

EXPECTED_ERRORS = 2


def test_two_invalid_ports_reported_in_one_module(tmp_path: Path) -> None:
    (tmp_path / "model.py").write_text(
        textwrap.dedent("""
        from typing import Annotated
        from bilt import Implementation, Interface, field, system, transform
        from bilt.units import Quantity
        from bilt.units.dimensions import LENGTH, MASS, TIME

        Length = Annotated[Quantity, LENGTH]
        Mass = Annotated[Quantity, MASS]
        Time = Annotated[Quantity, TIME]

        @system
        class Model(metaclass=Interface):
            a: Length = field()
            b: Length = field()

        @transform
        def a() -> Mass:
            return 0

        @transform
        def b() -> Time:
            return 0

        impl = Implementation(Model, [a, b])
        """),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text('[tool.bilt.check]\nmodules = ["model"]\n', encoding="utf-8")
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
    assert result.stderr.count("BILT002") == EXPECTED_ERRORS
    assert "port 'a'" in result.stderr
    assert "port 'b'" in result.stderr
    assert "2 errors found" in result.stderr
