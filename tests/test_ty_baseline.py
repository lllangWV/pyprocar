"""The typecheck gate, .github/scripts/ty_baseline.py, run against a scratch project.

It must fail on a diagnostic the baseline does not hold and on a baseline entry ty no
longer reports, and must pass when an edit only moves a baselined diagnostic.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / ".github" / "scripts" / "ty_baseline.py"

ONE_ERROR = 'def f() -> int:\n    return "a"\n'


@pytest.fixture
def project(tmp_path):
    (tmp_path / "ty.toml").write_text('[environment]\npython-version = "3.12"\n')
    (tmp_path / "mod.py").write_text(ONE_ERROR)
    return tmp_path


def gate(project: Path, *args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCRIPT), *args]
    return subprocess.run(cmd, cwd=project, capture_output=True, text=True)


def baseline(project: Path) -> dict:
    return json.loads((project / ".ty" / "baseline.json").read_text())


def test_regenerate_records_each_diagnostic_without_its_line(project):
    assert gate(project, "--regenerate").returncode == 0

    assert baseline(project)["files"] == {
        "mod.py": [
            {
                "rule": "invalid-return-type",
                "message": "Return type does not match returned value: "
                'expected `int`, found `Literal["a"]`',
            }
        ]
    }


def test_a_moved_diagnostic_passes(project):
    gate(project, "--regenerate")
    (project / "mod.py").write_text("import os\n\n\n" + ONE_ERROR)

    result = gate(project)

    assert result.returncode == 0, result.stdout + result.stderr


def test_a_new_diagnostic_fails_and_is_named_with_its_line(project):
    gate(project, "--regenerate")
    (project / "mod.py").write_text(ONE_ERROR + 'def g() -> int:\n    return "b"\n')

    result = gate(project)

    assert result.returncode == 1
    assert (
        "mod.py:4:12: invalid-return-type: Return type does not match returned value: "
        'expected `int`, found `Literal["b"]`'
    ) in result.stdout
    assert "1 new, 0 stale" in result.stdout


def test_a_stale_entry_fails_and_prune_deletes_only_it(project):
    (project / "mod.py").write_text(ONE_ERROR + 'def g() -> int:\n    return "b"\n')
    gate(project, "--regenerate")
    (project / "mod.py").write_text(ONE_ERROR)

    result = gate(project)

    assert result.returncode == 1
    assert "0 new, 1 stale" in result.stdout
    assert gate(project, "--prune").returncode == 0
    assert [e["message"][-14:] for e in baseline(project)["files"]["mod.py"]] == ['`Literal["a"]`']
    assert gate(project).returncode == 0


def test_a_baseline_from_another_ty_version_is_refused(project):
    gate(project, "--regenerate")
    data = baseline(project)
    data["ty"] = "0.0.1"
    (project / ".ty" / "baseline.json").write_text(json.dumps(data))

    result = gate(project)

    assert result.returncode == 1
    assert "was written by ty 0.0.1" in result.stderr


def test_an_absolute_path_matches_the_baseline_entries_below_it(project):
    gate(project, "--regenerate")

    result = gate(project, str(project / "mod.py"))

    assert result.returncode == 0, result.stdout + result.stderr


def test_a_path_ty_does_not_check_fails(project):
    (project / "ty.toml").write_text('[environment]\npython-version = "3.12"\n[src]\ninclude = ["mod.py"]\n')
    (project / "docs").mkdir()
    (project / "docs" / "conf.py").write_text(ONE_ERROR)
    gate(project, "--regenerate")

    result = gate(project, "docs")

    assert result.returncode == 1
    assert "No python files found" in result.stderr
