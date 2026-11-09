"""Verify ty and Pyrefly reject intentionally invalid quantity programs.

The source fixture is kept as .py.txt so normal repository-wide type checks
stay green. This test invokes each checker on a temporary Python copy and
matches error locations *and kinds*, not just the command's exit status.
"""

import pathlib
import re
import shutil
import subprocess
import tempfile

import pytest

FIXTURE = pathlib.Path(__file__).with_name("negative_quantity_cases.py.txt")
ROOT = pathlib.Path(__file__).resolve().parents[2]
MARKER = re.compile(r"# expect: ([a-z][a-z0-9-]+)")

# The two tools sometimes describe the same typing rule differently.
# Keep this mapping narrow: incorrect or unrelated diagnostics must fail.
EXPECTED_CODES: dict[str, dict[str, frozenset[str]]] = {
    "unlabelled-tensor-not-polymorphic": {
        "ty": frozenset({"invalid-argument-type"}),
        "pyrefly": frozenset({"bad-argument-type"}),
    },
    "polymorphic-has-no-tensor-reduction": {
        "ty": frozenset({"unresolved-attribute"}),
        "pyrefly": frozenset({"missing-attribute"}),
    },
    "unlabelled-tensor-is-not-a-quantity": {
        "ty": frozenset({"invalid-argument-type"}),
        "pyrefly": frozenset({"bad-argument-type"}),
    },
    "tensor-is-not-a-scalar-quantity": {
        "ty": frozenset({"invalid-argument-type"}),
        "pyrefly": frozenset({"bad-argument-type"}),
    },
    "scalar-is-not-a-tensor-quantity": {
        "ty": frozenset({"invalid-argument-type"}),
        "pyrefly": frozenset({"bad-argument-type"}),
    },
    "scalar-quantity-has-no-tensor-reduction": {
        "ty": frozenset({"unresolved-attribute"}),
        "pyrefly": frozenset({"missing-attribute"}),
    },
    "generic-quantity-has-no-arithmetic": {
        "ty": frozenset({"unsupported-operator"}),
        "pyrefly": frozenset({"unsupported-operation"}),
    },
    "unitless-tensor-is-not-quantity-tensor": {
        "ty": frozenset({"invalid-argument-type"}),
        "pyrefly": frozenset({"bad-argument-type"}),
    },
    "scalar-system-rejects-tensor": {
        "ty": frozenset({"invalid-argument-type"}),
        "pyrefly": frozenset({"bad-argument-type"}),
    },
    "tensor-system-rejects-scalar": {
        "ty": frozenset({"invalid-argument-type"}),
        "pyrefly": frozenset({"bad-argument-type"}),
    },
    "unitless-union-rejects-scalar-quantity": {
        "ty": frozenset({"invalid-argument-type"}),
        "pyrefly": frozenset({"bad-argument-type"}),
    },
}


def _marked_lines(source: str) -> dict[int, str]:
    marked = {
        number: match.group(1)
        for number, line in enumerate(source.splitlines(), start=1)
        if (match := MARKER.search(line)) is not None
    }
    assert marked
    assert set(marked.values()) == EXPECTED_CODES.keys()
    return marked


def _diagnostics(output: str, checker: str, filename: str) -> dict[int, set[str]]:
    found: dict[int, set[str]] = {}
    if checker == "ty":
        # ty --output-format concise: path:line:column: error[rule] ...
        regex = re.compile(rf"(?m)^.*{re.escape(filename)}:(\d+):\d+: error\[([a-z-]+)\]")
        matches = ((int(m.group(1)), m.group(2)) for m in regex.finditer(output))
    else:
        # pyrefly --output-format min-text: ERROR path:line:column-... [rule]
        regex = re.compile(rf"(?m)^ERROR .*{re.escape(filename)}:(\d+):\d+[^\n]*\[([a-z-]+)\]")
        matches = ((int(m.group(1)), m.group(2)) for m in regex.finditer(output))
    for line, code in matches:
        found.setdefault(line, set()).add(code)
    return found


@pytest.mark.parametrize("checker", ("ty", "pyrefly"))
def test_invalid_quantities_are_rejected_with_expected_diagnostics(checker: str) -> None:
    source = FIXTURE.read_text(encoding="utf-8")
    expected = _marked_lines(source)
    executable = shutil.which(checker)
    assert executable is not None, f"{checker} must be installed in the dev environment"

    with tempfile.TemporaryDirectory(prefix="negative-quantity-", dir=FIXTURE.parent) as directory:
        target = pathlib.Path(directory) / "cases.py"
        target.write_text(source, encoding="utf-8")
        output_format = "concise" if checker == "ty" else "min-text"
        result = subprocess.run(  # noqa: S603 - known installed checkers; no shell
            [executable, "check", "--output-format", output_format, str(target)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
        output = result.stdout + "\n" + result.stderr
        actual = _diagnostics(output, checker, target.name)

    expected_locations = set(expected)
    actual_locations = set(actual)
    assert result.returncode != 0, f"{checker} incorrectly accepted the invalid fixture"
    assert actual_locations == expected_locations, (
        f"{checker}: expected only errors on lines {sorted(expected_locations)}, "
        f"got lines {sorted(actual_locations)}\n{output}"
    )
    for line, label in expected.items():
        assert actual[line] & EXPECTED_CODES[label][checker], (
            f"{checker}: unexpected diagnostic for {label} at line {line}: {actual[line]}\n{output}"
        )
