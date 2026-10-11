"""Every module in pyprocar/ imports.

A package ``__init__.py`` imports only some of its submodules, so ``import pyprocar``
can pass while a submodule a user imports directly raises. pyprocar/scripts/scriptVector.py
imported a name pyprocar/utils/__init__.py no longer exported and failed for anyone who
imported it (issue #62). This test imports each module under pyprocar/ by name.
"""

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "pyprocar"


def _module_name(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


MODULES = sorted(
    _module_name(path)
    for path in PACKAGE.rglob("*.py")
    if not any(part.startswith(".") for part in path.relative_to(ROOT).parts)
)


@pytest.mark.guards_existing_behaviour(
    reason="an empty module list would make test_module_imports pass with no cases"
)
def test_finds_the_package_modules():
    assert "pyprocar.utils.utilsprocar" in MODULES
    assert "pyprocar.scripts" in MODULES


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str):
    importlib.import_module(module)
