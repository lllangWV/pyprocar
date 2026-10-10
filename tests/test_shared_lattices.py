"""Geometry tests take their cells and changes of basis from ``tests/utils/lattices.py``.

A test that builds its own two or three lattices never meets the cells that broke #302: the fcc
primitive cell, the acute Bi2Se3 rhombohedral cell, a row permutation, a sign flip and 1e-16 of
rounding noise. This check reads the source of every module under ``tests/`` and fails on

- a 3x3 literal whose rows are, up to one factor, those of a cell of ``STANDARD_CELLS`` or of its
  reciprocal lattice. A multiple of the identity is left out: every cubic cell, a parser's
  included, is one;
- a 3x3 literal equal to one of ``SHEARS``, to the reciprocal of one (the cubic reciprocal
  lattice in that basis), or to ``TURN``;
- a function, class or module constant named like a helper the shared module replaced
  (``rhombohedral``, ``sheared``, ``STANDARD_CELLS``, ``SHEARS``, a crystal's
  ``*_reciprocal_lattice``) that builds a 3x3 literal of its own.

A literal counts where the test makes a matrix of it: as the first argument of ``np.array`` or
``np.asarray``, in arithmetic, or as the value assigned or returned. It is evaluated with the
calls and arithmetic around it and with the module constants it names, so ``2.04 *
np.array(rows)`` and ``ROWS * [a, a, c]`` count. A literal that equals a shared lattice but is
not a copy of one, such as the lattice a parser reads from a fixture, takes the comment
``# not-a-shared-lattice: <reason>`` on its first line.
"""

import ast
import itertools
import math
import re
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import pytest

from tests.utils.lattices import SHEARS, STANDARD_CELLS, TURN, reciprocal

ROOT = Path(__file__).resolve().parent.parent
SHARED = ROOT / "tests" / "utils" / "lattices.py"
EXCEPTION = "# not-a-shared-lattice:"
HELPER_NAMES = re.compile(
    r"rhombohedral|sheared|STANDARD_CELLS|SHEARS"
    + r"|\w*(cubic|fcc|bcc|hex|hexagonal|rhombohedral|monoclinic|triclinic)\w*_reciprocal_lattice",
    re.IGNORECASE,
)
FUNCTIONS = {"sqrt": math.sqrt, "radians": math.radians, "cos": math.cos, "sin": math.sin}
ARRAY_CALLS = {"array", "asarray"}
MATRIX_CONTEXTS = (ast.BinOp, ast.UnaryOp, ast.Assign, ast.AnnAssign, ast.Return)
TOLERANCE = 1e-6


def _shown(path: Path) -> Path:
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def _shared_lattices() -> list[tuple[str, np.ndarray]]:
    """(name in the shared module, rows) of every cell and its reciprocal lattice, but cubic."""
    found = []
    for name, cell in STANDARD_CELLS.items():
        if not np.array_equal(cell, cell[0, 0] * np.eye(3)):
            found.append((f'STANDARD_CELLS["{name}"]', cell))
            found.append((f'reciprocal(STANDARD_CELLS["{name}"])', reciprocal(cell)))
    return found


def _shared_transforms() -> list[tuple[str, np.ndarray]]:
    found = [("TURN", TURN)]
    for name, change in SHEARS.items():
        found.append((f'SHEARS["{name}"]', change))
        found.append((f'reciprocal(SHEARS["{name}"])', reciprocal(change)))
    return found


LATTICES = _shared_lattices()
TRANSFORMS = _shared_transforms()


class _NotConstant(Exception):
    pass


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return node.func.id if isinstance(node.func, ast.Name) else None


class _Evaluator:
    """Numbers, nested lists, ``np.array``, arithmetic, sqrt and pi, and module constants."""

    def __init__(self, tree: ast.Module):
        self.constants: dict[str, ast.expr] = {
            node.targets[0].id: node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }

    def __call__(self, node: ast.expr, seen: frozenset[str] | None = None) -> np.ndarray:
        seen = frozenset() if seen is None else seen
        match node:
            case ast.Constant(value=bool()):
                raise _NotConstant
            case ast.Constant(value=int() | float() as value):
                return np.asarray(float(value))
            case ast.List(elts=elts) | ast.Tuple(elts=elts):
                values = [self(element, seen) for element in elts]
                if len({value.shape for value in values}) > 1:
                    raise _NotConstant
                return np.asarray(values, dtype=np.float64)
            case ast.UnaryOp(op=ast.USub(), operand=operand):
                return -self(operand, seen)
            case ast.UnaryOp(op=ast.UAdd(), operand=operand):
                return self(operand, seen)
            case ast.BinOp(left=left, op=op, right=right):
                return _arithmetic(op, self(left, seen), self(right, seen))
            case ast.Attribute(attr="pi"):
                return np.asarray(math.pi)
            case ast.Name(id=name) if name in self.constants and name not in seen:
                return self(self.constants[name], seen | {name})
            case ast.Call(args=[argument, *_]) if _call_name(node) in ARRAY_CALLS:
                return self(argument, seen)
            case ast.Call(args=[argument]) if _call_name(node) in FUNCTIONS:
                function = FUNCTIONS[str(_call_name(node))]
                return np.vectorize(function)(self(argument, seen))
            case _:
                raise _NotConstant


def _arithmetic(op: ast.operator, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    with np.errstate(all="raise"):
        try:
            match op:
                case ast.Add():
                    return left + right
                case ast.Sub():
                    return left - right
                case ast.Mult():
                    return left * right
                case ast.Div():
                    return left / right
                case ast.Pow():
                    return left**right
                case _:
                    raise _NotConstant
        except (ValueError, FloatingPointError) as error:
            raise _NotConstant from error


def _is_3x3_display(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.List | ast.Tuple)
        and len(node.elts) == 3
        and all(isinstance(row, ast.List | ast.Tuple) and len(row.elts) == 3 for row in node.elts)
    )


def _is_matrix(node: ast.AST, parent: Mapping[ast.AST, ast.AST]) -> bool:
    """A 3x3 display the test makes a matrix of, not a list of three index triples."""
    outer = parent.get(node)
    if isinstance(outer, ast.Call):
        return _call_name(outer) in ARRAY_CALLS and bool(outer.args) and outer.args[0] is node
    return isinstance(outer, MATRIX_CONTEXTS)


def _scale(matrix: np.ndarray, rows: np.ndarray) -> float | None:
    """The factor s with matrix = s rows, if there is one."""
    scale = float((matrix * rows).sum() / (rows * rows).sum())
    close = np.abs(matrix - scale * rows).max() <= TOLERANCE * np.abs(matrix).max()
    return scale if close and abs(scale) > TOLERANCE else None


def _shared_name(matrix: np.ndarray) -> str | None:
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not matrix.any():
        return None
    for name, rows in TRANSFORMS:
        if np.allclose(matrix, rows, rtol=0, atol=TOLERANCE):
            return name
    for name, rows in LATTICES:
        scale = _scale(matrix, rows)
        if scale is not None:
            return name if math.isclose(scale, 1.0) else f"{scale:.6g} * {name}"
    return None


def _values(
    node: ast.expr, parent: Mapping[ast.AST, ast.AST], evaluate: _Evaluator
) -> Iterator[np.ndarray]:
    """The expression's value, then that of each enclosing call or arithmetic that evaluates."""
    while True:
        try:
            yield evaluate(node)
        except _NotConstant:
            return
        outer = parent.get(node)
        if not isinstance(outer, ast.Call | ast.BinOp | ast.UnaryOp):
            return
        node = outer


def _literal_copies(
    tree: ast.Module, lines: list[str], parent: Mapping[ast.AST, ast.AST]
) -> Iterator[tuple[int, str]]:
    evaluate = _Evaluator(tree)
    flagged: set[ast.AST] = set()

    def inside_flagged(node: ast.AST) -> bool:
        while node in parent:
            node = parent[node]
            if node in flagged:
                return True
        return False

    # Breadth first, so arithmetic is judged before the literal it holds.
    for node in ast.walk(tree):
        # A matrix literal, or arithmetic on a constant that holds one, as in ROWS * [a, a, c].
        matrix = _is_3x3_display(node) and _is_matrix(node, parent)
        if not (matrix or isinstance(node, ast.BinOp)) or inside_flagged(node):
            continue
        assert isinstance(node, ast.expr)
        if EXCEPTION in lines[node.lineno - 1]:
            continue
        name = next(filter(None, map(_shared_name, _values(node, parent, evaluate))), None)
        if name is not None:
            flagged.add(node)
            yield node.lineno, name


def _helper_copies(tree: ast.Module) -> Iterator[tuple[int, str]]:
    scopes = [tree.body, *(n.body for n in tree.body if isinstance(n, ast.ClassDef))]
    for statement in itertools.chain.from_iterable(scopes):
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names = [statement.name]
        elif isinstance(statement, ast.Assign):
            names = [t.id for t in statement.targets if isinstance(t, ast.Name)]
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            names = [statement.target.id]
        else:
            continue
        builds_one = any(_is_3x3_display(node) for node in ast.walk(statement))
        for name in names:
            if builds_one and not name.startswith("test") and HELPER_NAMES.fullmatch(name):
                yield statement.lineno, name


def private_lattices(paths: list[Path]) -> list[str]:
    found = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, str(path))
        parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for line, name in _literal_copies(tree, source.splitlines(), parent):
            found.append(
                f"{_shown(path)}:{line}: this 3x3 literal is {name}; import it from"
                + " tests/utils/lattices.py, and check a basis-dependent result over"
                + " basis_variants of every cell in STANDARD_CELLS"
            )
        for line, name in _helper_copies(tree):
            found.append(
                f"{_shown(path)}:{line}: {name} builds its own lattice; import the cell,"
                + " rhombohedral, shear, SHEARS, STANDARD_CELLS or basis_variants from"
                + " tests/utils/lattices.py"
            )
    return found


def _test_files() -> list[Path]:
    return sorted(path for path in (ROOT / "tests").rglob("*.py") if path != SHARED)


@pytest.mark.guards_existing_behaviour(
    reason="it scans tests/, which this PR already migrates; the self-test proves each form fails"
)
def test_tests_take_their_lattices_from_the_shared_module():
    assert private_lattices(_test_files()) == []


def _line_numbers(hits: list[str]) -> list[int]:
    return sorted(int(hit.split(": ")[0].rsplit(":", 1)[1]) for hit in hits)


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.mark.guards_existing_behaviour(
    reason="it runs the check on synthetic files and never imports pyprocar"
)
def test_the_check_flags_the_lattices_tests_built_before_the_shared_module(tmp_path):
    """The true positives are the definitions dev's #302 tests made for themselves, in
    test_fermisurface_drawn_mesh.py, test_cross_section_area.py and test_brillouin_zone.py."""
    drawn_mesh = _write(
        tmp_path / "test_drawn_mesh.py",
        [
            "import math",
            "import numpy as np",
            "import pytest",
            "CUBIC = np.eye(3)",
            "U1 = np.array([[0, -1, 4], [2, -2, -5], [1, -2, 2]], dtype=float)",
            "HEX_REAL = np.array([[1.0, 0, 0], [-0.5, math.sqrt(3) / 2, 0], [0, 0, 1.6]])",
            "FCC = np.linalg.inv(np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])).T",
            "BCC = np.linalg.inv(0.5 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])).T",
            "def sheared(k: int) -> np.ndarray:",
            "    return np.array([[1.0, 0, 0], [k, 1, 0], [0, 0, 1]])",
            "@pytest.mark.parametrize(",
            '    "lattice", [U1, np.array([[1.0, 0, 0], [3, 1, 0], [-2, 2, 1]])], ids=["U1", "b"]',
            ")",
            "def test_general_shear(lattice):",
            "    pass",
            "def rhombohedral(a: float, alpha: float) -> np.ndarray:",
            "    c = math.cos(math.radians(alpha))",
            "    tx, ty, tz = math.sqrt((1 - c) / 2), math.sqrt((1 - c) / 6), 1.0",
            "    return a * np.array([[tx, -ty, tz], [0, 2 * ty, tz], [-tx, -ty, tz]])",
            "TURN = np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]])",
        ],
    )
    cross_section = _write(
        tmp_path / "test_cross_section.py",
        [
            "import numpy as np",
            "A3_PLUS_6_A1 = np.array([[1.0, 0.0, -6.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])",
            "_HEX_ROWS = np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.0]])",
            "BI2SE3_RHOMBOHEDRAL = np.array(",
            "    [",
            "        [2.068734, -1.194384, 9.546678],",
            "        [0.0, 2.388768, 9.546678],",
            "        [-2.068734, -1.194384, 9.546678],",
            "    ]",
            ")",
            "STANDARD_CELLS = {",
            '    "SrVO3 cubic": 3.84652 * np.eye(3),',
            '    "Au fcc, van-alphen POSCAR": 2.949546',
            "    * np.array([[1.0, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0.5, np.sqrt(3) / 6,"
            + " np.sqrt(2 / 3)]]),",
            '    "fcc primitive": 2.04 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]),',
            '    "bcc primitive": 1.65 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]]),',
            '    "hexagonal c/a 2.72": _HEX_ROWS * [2.46, 2.46, 6.7],',
            '    "hexagonal c/a 1.62": _HEX_ROWS * [3.21, 3.21, 5.21],',
            '    "Bi2Se3 rhombohedral": BI2SE3_RHOMBOHEDRAL,',
            "}",
            "SHEARS = {",
            '    "given": np.eye(3, dtype=int),',
            '    "a3 + 6 a1": np.array([[1, 0, 0], [0, 1, 0], [6, 0, 1]]),',
            '    "doubly sheared": np.array([[1, 0, 0], [3, 1, 0], [3, 3, 1]]),',
            "}",
            "def test_fcc_plane():",
            "    fcc = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) / 4.08",
        ],
    )
    brillouin_zone = _write(
        tmp_path / "test_brillouin_zone.py",
        [
            "import numpy as np",
            "import pytest",
            "@pytest.fixture",
            "def fcc_reciprocal_lattice():",
            "    a = 2 * np.pi",
            "    return np.array([",
            "        [-a, a, a],",
            "        [a, -a, a],",
            "        [a, a, -a]",
            "    ])",
            "MONOCLINIC = np.array([[1.0, 0.0, 0.0], [0.0, 1.3, 0.0], [0.4, 0.0, 0.9]])",
        ],
    )

    hits = private_lattices([drawn_mesh, cross_section, brillouin_zone])

    def lines(path: Path) -> list[int]:
        return _line_numbers([hit for hit in hits if hit.startswith(f"{_shown(path)}:")])

    assert lines(drawn_mesh) == [5, 7, 8, 9, 12, 16, 20]
    assert lines(cross_section) == [2, 5, 11, 13, 15, 16, 17, 18, 21, 23, 24, 27]
    assert lines(brillouin_zone) == [4, 11]
    assert (
        f'{_shown(drawn_mesh)}:7: this 3x3 literal is 0.245098 * STANDARD_CELLS["fcc primitive"];'
        + " import it from tests/utils/lattices.py"
    ) in "\n".join(hits)
    assert (
        f"{_shown(drawn_mesh)}:9: sheared builds its own lattice; import the cell, rhombohedral,"
    ) in "\n".join(hits)


@pytest.mark.guards_existing_behaviour(
    reason="it runs the check on synthetic files and never imports pyprocar"
)
def test_the_check_passes_parser_lattices_transforms_and_index_lists(tmp_path):
    """A parser's lattice, an identity, a cubic cell, index triples, a fixture or helper that
    takes its lattice from the shared module, and a marked exception are not copies."""
    clean = _write(
        tmp_path / "test_clean.py",
        [
            "import numpy as np",
            "import pytest",
            "from tests.utils.lattices import SHEARS, fcc_primitive, reciprocal, shear",
            "GRAPHENE = np.array(",
            "    [[-0.405231703, -0.233960597, 0.0], [-0.405231703, 0.233960597, 0.0],"
            + " [0.0, 0.0, -0.115140544]]",
            ")",
            "IDENTITY = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])",
            "CUBIC_2PI = np.array([[2 * np.pi, 0, 0], [0, 2 * np.pi, 0], [0, 0, 2 * np.pi]])",
            "SI_FROM_FDF = np.array(",
            "    [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]"
            + "  # not-a-shared-lattice: what the parser reads from FDF_STR",
            ")",
            "SI_AS_SIESTA_PRINTS_IT = np.array(",
            "    [[2.715, 2.715, 0], [0, 2.715, 2.715], [2.715, 0, 2.715]]",
            ")",
            "@pytest.mark.parametrize('signs', [(-1, 1, 1), (1, -1, 1), (1, 1, -1)])",
            "def test_signs(signs):",
            "    place(fractional_coordinates=[[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])",
            "@pytest.fixture",
            "def fcc_reciprocal_lattice():",
            "    return 2 * np.pi * reciprocal(fcc_primitive(1.0))",
            "SHEARED = SHEARS['U1'] @ np.eye(3)",
            "def sheared(k):",
            "    return shear(k) @ np.eye(3)",
        ],
    )

    assert private_lattices([clean]) == []

    marked = clean.read_text(encoding="utf-8").replace("  # not-a-shared-lattice:", "  #")
    assert _line_numbers(private_lattices([_write(clean, [marked])])) == [10]
