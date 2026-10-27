import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_every_tool_checks_the_declared_python_floor():
    requires = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "requires-python"
    ]
    floor = re.fullmatch(r">=(3\.\d+)", requires)
    assert floor, f"pyproject.toml requires-python {requires!r} is not of the form >=3.N"
    major_minor = floor.group(1)

    pixi = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))["dependencies"]["python"]
    ruff = tomllib.loads((ROOT / ".config/.ruff.toml").read_text(encoding="utf-8"))[
        "target-version"
    ]
    ty = tomllib.loads((ROOT / "ty.toml").read_text(encoding="utf-8"))["environment"][
        "python-version"
    ]

    assert pixi.startswith(f">={major_minor},"), f"pixi.toml python {pixi!r} != {requires}"
    ruff_floor = "py" + major_minor.replace(".", "")
    assert ruff == ruff_floor, f".config/.ruff.toml target-version {ruff!r} != {requires}"
    assert ty == major_minor, f"ty.toml environment.python-version {ty!r} != {requires}"
    docs = re.search(
        r'^\s+python: "(3\.\d+)"', (ROOT / ".readthedocs.yaml").read_text(encoding="utf-8"), re.M
    )
    assert docs and docs.group(1) == major_minor, f".readthedocs.yaml build python != {requires}"
