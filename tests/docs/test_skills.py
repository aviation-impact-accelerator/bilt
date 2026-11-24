import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / ".agents" / "skills"


def test_modelling_skill_examples(tmp_path: Path) -> None:
    skill = (SKILLS / "bilt" / "SKILL.md").read_text(encoding="utf-8")
    examples = re.findall(r"^```python\n(.*?)^```$", skill, flags=re.MULTILINE | re.DOTALL)
    assert examples

    for index, example in enumerate(examples):
        source = tmp_path / f"skill_example_{index}.py"
        source.write_text(example, encoding="utf-8")
        runpy.run_path(str(source))


def test_development_skill_checks_match_ci() -> None:
    script = (SKILLS / "bilt-development" / "scripts" / "check.sh").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "check.yml").read_text(encoding="utf-8")

    # Compare the checks, not GitHub-specific environment setup or artifact uploads.
    local_checks = re.findall(r"^run ((?:uv |cargo ).+)$", script, flags=re.MULTILINE)
    ci_checks = re.findall(r"^ +run: ((?:uv |cargo ).+)$", workflow, flags=re.MULTILINE)

    assert ci_checks
    assert local_checks == ci_checks
