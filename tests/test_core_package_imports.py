"""Code inside pyprocar/ imports core names from the submodule that defines them.

pyprocar/core/__init__.py re-exports the core classes for users. A parser that imports
through it (``from pyprocar.core import Structure``) closes an import cycle through that
file, because core/ebs.py and core/dos.py call the parsers. basedpyright reports at most
four cycles per file, and which four depends on the import graph, so a new import
anywhere in pyprocar.core added or removed ``reportImportCycles`` baseline entries on
pyprocar/core/__init__.py and failed lock mode on code nobody changed (issue #60).
This test fails on any import of the package itself from pyprocar/, and names the
submodule to import from instead.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "pyprocar"
CORE_INIT = PACKAGE / "core" / "__init__.py"
THROUGH_PACKAGE = "imports {name} through pyprocar/core/__init__.py; use `{instead}`"
WHOLE_PACKAGE = (
    "imports the package pyprocar.core, which loads pyprocar/core/__init__.py; "
    + "import the submodule that defines the name"
)


OWNERS = {
    alias.asname or alias.name: node.module
    for node in ast.parse(CORE_INIT.read_text(encoding="utf-8")).body
    if isinstance(node, ast.ImportFrom) and node.module
    for alias in node.names
}


def _absolute(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    package = parts[: len(parts) - node.level]
    return ".".join(package + ([node.module] if node.module else []))


def _instead(name: str) -> str:
    if name in OWNERS:
        return f"from {OWNERS[name]} import {name}"
    return f"import pyprocar.core.{name} as {name}"


def core_package_imports(path: Path, source: str) -> list[str]:
    where = path.relative_to(ROOT)
    found = []
    for node in ast.walk(ast.parse(source, str(path))):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = _absolute(path, node)
            modules = [f"{module}.{alias.name}" for alias in node.names]
            if module == "pyprocar.core":
                found += [
                    f"{where}:{node.lineno}: "
                    + THROUGH_PACKAGE.format(name=alias.name, instead=_instead(alias.name))
                    for alias in node.names
                ]
                continue
        else:
            continue
        found += [
            f"{where}:{node.lineno}: {WHOLE_PACKAGE}"
            for module in modules
            if module == "pyprocar.core"
        ]
    return found


def test_library_code_never_imports_the_core_package():
    found = [
        message
        for path in sorted(PACKAGE.rglob("*.py"))
        if path != CORE_INIT
        for message in core_package_imports(path, path.read_text(encoding="utf-8"))
    ]
    assert found == []


@pytest.mark.guards_existing_behaviour(reason="it runs the check on synthetic sources")
def test_every_import_form_of_the_core_package_is_caught():
    path = PACKAGE / "io" / "example.py"
    first = "pyprocar/io/example.py:1: "
    sources = {
        "from pyprocar.core import Structure, kpoints": [
            first
            + "imports Structure through pyprocar/core/__init__.py; "
            + "use `from pyprocar.core.structure import Structure`",
            first
            + "imports kpoints through pyprocar/core/__init__.py; "
            + "use `import pyprocar.core.kpoints as kpoints`",
        ],
        "def f():\n    from ..core import KPath": [
            "pyprocar/io/example.py:2: imports KPath through pyprocar/core/__init__.py; "
            + "use `from pyprocar.core.kpoints import KPath`",
        ],
        "import pyprocar.core": [first + WHOLE_PACKAGE],
        "from pyprocar import core": [first + WHOLE_PACKAGE],
        "from pyprocar.core.structure import Structure\nimport pyprocar.core.kpoints as kp": [],
    }
    for source, expected in sources.items():
        assert core_package_imports(path, source) == expected, source
