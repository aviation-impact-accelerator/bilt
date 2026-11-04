"""Run generation, runtime tests and positive/negative checks with all three checkers."""

from pathlib import Path
import os
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BIN = Path(sys.executable).parent
ENV = os.environ | {
    "VIRTUAL_ENV": sys.prefix,
    "PATH": str(BIN) + os.pathsep + os.environ.get("PATH", ""),
    "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(HERE))),
    "MYPYPATH": os.pathsep.join((str(ROOT / "src"), str(HERE))),
}


def run(command):
    result = subprocess.run(command, cwd=ROOT, env=ENV, text=True, capture_output=True)
    return result.returncode, result.stdout + result.stderr


def require(command):
    code, output = run(command)
    print(output, end="")
    if code:
        raise SystemExit(code)


require([sys.executable, str(HERE / "generate.py"), "--check"])
require([sys.executable, "-m", "pytest", "-q", "-c", "/dev/null", str(HERE / "tests")])
positive = [
    str(HERE / "examples.py"),
    str(HERE / "typing_cases" / "valid.py"),
    str(HERE / "bilt_typing" / "_values.pyi"),
    str(HERE / "bilt_typing" / "_bridge.pyi"),
]
negative = HERE / "typing_cases" / "invalid.py"
expected = {i for i, line in enumerate(negative.read_text().splitlines(), 1) if "# error" in line}
commands = {
    "mypy": [str(BIN / "mypy"), "--follow-imports=silent", "--python-version", "3.15"],
    "pyrefly": [
        str(BIN / "pyrefly"),
        "check",
        "--preset",
        "strict",
        "--python-interpreter-path",
        sys.executable,
        "--search-path",
        str(HERE),
        "--search-path",
        str(ROOT / "src"),
    ],
    "ty": [str(BIN / "ty"), "check", "--extra-search-path", str(HERE), "--extra-search-path", str(ROOT / "src")],
}
for checker, command in commands.items():
    require(command + positive)
    code, output = run(command + [str(negative)])
    actual = {int(n) for n in re.findall(r"invalid\.py:(\d+):", output)}
    if code == 0 or actual != expected:
        print(output)
        raise SystemExit(f"{checker}: expected errors on {sorted(expected)}, got {sorted(actual)}")
    print(f"{checker}: all {len(expected)} invalid examples rejected")
print("Prototype checks passed.")
