"""The operations VASP reduced its k-mesh with, when the OUTCAR prints no operator table.

VASP prints the table only with NWRITE = 3. The default NWRITE = 2 leaves the point group's
name, the operation count and the irreducible k-points, and pyprocar recovers the operations
from those (issue #5).
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pyprocar.core.dos import DensityOfStates
from pyprocar.io.vasp.outcar import Outcar
from pyprocar.io.vasp.parser import VaspParser
from tests.utils import DATA_DIR

EXAMPLES = DATA_DIR / "examples"

# A one-atom monoclinic cell, unique axis b: the crystal's point group is 2/m (C_2h).
LATTICE = np.array([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [-1.0, 0.0, 6.0]])
RECIPROCAL = np.array([[0.25, 0.0, 1 / 24], [0.0, 0.2, 0.0], [0.0, 0.0, 1 / 6]])

# k-space rotations of 2/m in this cell: a 2-fold axis along b turns a and c over.
E = np.eye(3, dtype=int)
INVERSION = -E
C2_B = np.diag([-1, 1, -1])


@dataclass(frozen=True)
class Mesh:
    kind: str
    size: tuple[int, int, int]
    shift: tuple[float, float, float]  # as VASP prints it: Gamma's offset, in mesh steps


GAMMA_331 = Mesh("Gamma", (3, 3, 1), (0.0, 0.0, 0.0))
# An even Monkhorst-Pack mesh: VASP 6.5.0 prints the shift 0.5 along each even axis.
MONKHORST_PACK_221 = Mesh("Monkhorst-Pack", (2, 2, 1), (0.5, 0.5, 0.0))

THIRD = 1 / 3
# Gamma 3x3x1 reduced by {E, I}, which time reversal alone also gives: each k pairs with -k.
PAIRED_WITH_MINUS_K = [
    ((0, 0, 0), 1),
    ((THIRD, 0, 0), 2),
    ((0, THIRD, 0), 2),
    ((THIRD, THIRD, 0), 2),
    ((-THIRD, THIRD, 0), 2),
]
# Monkhorst-Pack 2x2x1, k = (+-1/4, +-1/4, 0), reduced by {E, I}.
QUARTERS_PAIRED_WITH_MINUS_K = [((0.25, 0.25, 0), 2), ((-0.25, 0.25, 0), 2)]
# Gamma 3x3x1 reduced by {E, C2_b} alone: (k1, k2, 0) pairs with (-k1, k2, 0).
TURNED_ABOUT_B = [
    ((0, 0, 0), 1),
    ((THIRD, 0, 0), 2),
    ((0, THIRD, 0), 1),
    ((0, -THIRD, 0), 1),
    ((THIRD, THIRD, 0), 2),
    ((THIRD, -THIRD, 0), 2),
]

OVERALL_S2 = """
The overall configuration has the point symmetry S_2 .


 Subroutine INISYM returns: Found  2 space group operations
 (whereof  2 operations are pure point group operations),
 and found     1 'primitive' translations
"""
OVERALL_C2 = OVERALL_S2.replace("S_2", "C_2")
NOSYMM = "NOSYMM: (Re-)initialisation of all symmetry stuff for point group C_1.\n"


def _rows(matrix: np.ndarray) -> str:
    return "\n".join("  " + "  ".join(f"{x:13.9f}" for x in row) for row in matrix)


def synthetic_outcar(
    symmetry: str, irreducible, noncollinear: bool = False, mesh: Mesh = GAMMA_331
) -> str:
    """An NWRITE = 2 OUTCAR of the one-atom monoclinic cell."""
    lattices = "\n".join(
        "  " + "  ".join(f"{x:13.9f}" for x in (*a, *b)) for a, b in zip(LATTICE, RECIPROCAL)
    )
    listed = "\n".join(
        f"  {k[0]:9.6f} {k[1]:9.6f} {k[2]:9.6f}      {w:8.6f}" for k, w in irreducible
    )
    flag = "T" if noncollinear else "F"
    grid = "".join(f"{n:5d}" for n in mesh.size)
    shift = "  ".join(f"{s:.9f}" for s in mesh.shift)
    return f""" vasp.6.4.2 20Jul23 (build Apr  3 2024 16:30:41) complex
 POSCAR = monoclinic

Analysis of structural, dynamic, and magnetic symmetry:
=====================================================================
{symmetry}

 KPOINTS: {mesh.kind}

Automatic generation of k-mesh.
 Grid dimensions read from file:
 generate k-points for:{grid}

 Generating k-lattice:

  Shift w.r.t. Gamma in fractional coordinates (k-lattice)
     {shift}


 Subroutine IBZKPT returns following result:
 ===========================================

 Found {len(irreducible):6d} irreducible k-points:

 Following reciprocal coordinates:
            Coordinates               Weight
{listed}

 Dimension of arrays:
   k-points           NKPTS = {len(irreducible):6d}   k-points in BZ     NKDIM = {len(irreducible):6d}
   ions per type =               1
--------------------------------------------------------------------------------------------------------
   ISPIN  =      1    spin polarized calculation?
   LNONCOLLINEAR =      {flag} non collinear calculations

      direct lattice vectors                 reciprocal lattice vectors
{lattices}

  length of vectors

 position of ions in fractional coordinates (direct lattice)
   0.00000000  0.00000000  0.00000000

 E-fermi :   0.0000     XC(G=0):  -1.0000     alpha+bet : -1.0000

"""


POSCAR = f"""monoclinic
1.0
{_rows(LATTICE)}
Pb
1
Direct
  0.0  0.0  0.0
"""

def kpoints_file(mesh: Mesh = GAMMA_331) -> str:
    return f"{mesh.kind}\n0\n{mesh.kind}\n {' '.join(str(n) for n in mesh.size)}\n 0 0 0\n"


def synthetic_procar(irreducible) -> str:
    """One band whose energy is the irreducible k-point's 1-based index."""
    total = sum(weight for _, weight in irreducible)
    blocks = [
        f"PROCAR lm decomposed\n# of k-points:  {len(irreducible)}"
        + "         # of bands:    1         # of ions:    1\n"
    ]
    for index, (k, weight) in enumerate(irreducible, start=1):
        blocks.append(
            f"\n k-point {index:5d} :    {k[0]:.8f} {k[1]:.8f} {k[2]:.8f}"
            + f"     weight = {weight / total:.8f}\n\n"
            + f"band     1 # energy {index:13.8f} # occ.  1.00000000\n \n"
            + "ion      s     py     pz     px    dxy    dyz    dz2    dxz  x2-y2    tot\n"
            + "    1  1.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000\n \n"
        )
    return "".join(blocks)


def as_set(rotations) -> set[tuple[int, ...]]:
    return {tuple(np.rint(rotation).astype(int).ravel()) for rotation in rotations}


def test_reporter_shaped_outcar_recovers_inversion():
    """2/m has one subgroup named S_2 (Ci): {E, I}."""
    outcar = Outcar.from_str(synthetic_outcar(OVERALL_S2, PAIRED_WITH_MINUS_K))

    assert as_set(outcar.rotations) == as_set([E, INVERSION])


def test_noncollinear_outcar_reduced_without_time_reversal_recovers_the_twofold_axis():
    """VASP reduces a non-collinear mesh by its operations alone, without k -> -k.

    Only {E, C2_b} leaves (0, 1/3, 0) and (0, -1/3, 0) in separate orbits.
    """
    outcar = Outcar.from_str(synthetic_outcar(OVERALL_C2, TURNED_ABOUT_B, noncollinear=True))

    assert as_set(outcar.rotations) == as_set([E, C2_B])


def test_nosymm_outcar_recovers_the_identity():
    """ISYM = 0: VASP names C_1 and pairs each k with -k by time reversal alone."""
    outcar = Outcar.from_str(synthetic_outcar(NOSYMM, PAIRED_WITH_MINUS_K))

    assert as_set(outcar.rotations) == as_set([E])


@pytest.mark.parametrize(
    ("symmetry", "irreducible", "mesh"),
    [
        pytest.param(OVERALL_S2, PAIRED_WITH_MINUS_K, GAMMA_331, id="S_2-group"),
        pytest.param(NOSYMM, PAIRED_WITH_MINUS_K, GAMMA_331, id="ISYM=0"),
        pytest.param(
            OVERALL_S2, QUARTERS_PAIRED_WITH_MINUS_K, MONKHORST_PACK_221, id="Monkhorst-Pack"
        ),
    ],
)
def test_reporter_shaped_mesh_unfolds_to_the_full_grid(symmetry, irreducible, mesh):
    """Each grid point gets the energy of the irreducible k-point it pairs with, k or -k."""
    parser = VaspParser.from_str(
        outcar=synthetic_outcar(symmetry, irreducible, mesh=mesh),
        procar=synthetic_procar(irreducible),
        kpoints=kpoints_file(mesh),
        poscar=POSCAR,
    )

    ebs = parser.ebs

    assert ebs is not None
    assert ebs.n_kpoints == np.prod(mesh.size)
    sources = np.array([k for k, _ in irreducible])
    for kpoint, energy in zip(ebs.kpoints, ebs.bands[:, 0, 0], strict=True):
        steps = np.stack([kpoint - sources, kpoint + sources])
        paired = np.all(np.abs(steps - np.rint(steps)) < 1e-6, axis=2).any(axis=0)
        assert energy == np.flatnonzero(paired)[0] + 1, kpoint


@pytest.mark.guards_existing_behaviour(
    reason="main recovers no operations at all; this pins that the fit refuses this list"
)
def test_collinear_outcar_whose_list_needs_no_time_reversal_recovers_nothing():
    """A collinear run pairs k with -k, so no subgroup of 2/m gives the C_2 list."""
    outcar = Outcar.from_str(synthetic_outcar(OVERALL_C2, TURNED_ABOUT_B))

    assert len(outcar.rotations) == 0


def test_named_group_that_does_not_fit_the_list_raises_from_ebs():
    """With time reversal, {E, C2_b} sends (1/3, 1/3, 0) to 4 grid points, not the 2 listed."""
    parser = VaspParser.from_str(
        outcar=synthetic_outcar(OVERALL_C2, PAIRED_WITH_MINUS_K),
        procar=synthetic_procar(PAIRED_WITH_MINUS_K),
        kpoints=kpoints_file(),
        poscar=POSCAR,
    )

    assert len(parser.outcar.rotations) == 0
    with pytest.raises(ValueError, match=r"5 of the 9 k-points.*NWRITE = 3.*ISYM = -1"):
        _ = parser.ebs


# Every fixture OUTCAR that prints a VASP 6 operator table and reduces an automatic mesh.
TABLE_FIXTURES = [
    "dos/non-spin-polarized",
    "dos/spin-polarized",
    "dos/non-colinear",
    "fermi2d/non-spin-polarized",
    "fermi2d/spin-polarized",
    "fermi3d/non-spin-polarized",
    "fermi3d/spin-polarized",
    "fermi3d/non-colinear",
    "bands/2d-bands/graphene",
]


def strip_operator_table(text: str) -> str:
    """The OUTCAR as NWRITE = 2 writes it: no irot blocks."""
    return re.sub(r"^ irot\s*:.*?rotmap:\s*\n[^\n]*\n", "", text, flags=re.MULTILINE | re.DOTALL)


def printed_table(text: str) -> list[np.ndarray]:
    """VASP's isymop matrices, transposed: the k-space rotations VASP 6 lists."""
    rows = r"\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)"
    return [
        np.array(found, dtype=int).reshape(3, 3).T
        for found in re.findall(r"isymop:" + rows * 3, text)
    ]


@pytest.mark.data
@pytest.mark.parametrize("fixture", TABLE_FIXTURES)
def test_recovered_operations_equal_the_printed_table(fixture):
    text = (EXAMPLES / fixture / "OUTCAR").read_text()
    stripped = Outcar.from_str(strip_operator_table(text))
    table = printed_table(text)
    assert table, "the fixture prints an operator table"
    assert stripped.get_symmetry_operations() == []

    assert as_set(stripped.rotations) == as_set(table)


def _stripped_parser(fixture: Path) -> VaspParser:
    outcar = Outcar.from_str(strip_operator_table((fixture / "OUTCAR").read_text()))
    return VaspParser(dirpath=fixture, outcar=outcar)


@pytest.mark.data
def test_ambiguous_group_raises_from_ebs():
    """O_h has four S_6 subgroups, one per body diagonal, and the 61x61x1 list fits each."""
    parser = _stripped_parser(EXAMPLES / "fermi2d" / "non-colinear")

    assert parser.outcar is not None
    assert len(parser.outcar.rotations) == 0
    with pytest.raises(ValueError, match=r"1831 of the 3721 k-points.*NWRITE = 3.*ISYM = -1"):
        _ = parser.ebs


@pytest.mark.data
@pytest.mark.guards_existing_behaviour(
    reason="the DOS of an NWRITE = 2 run reads as before; only ebs raises when it cannot unfold"
)
def test_ambiguous_group_still_reads_the_dos():
    dos = _stripped_parser(EXAMPLES / "fermi2d" / "non-colinear").dos

    assert isinstance(dos, DensityOfStates)


@pytest.mark.data
@pytest.mark.guards_existing_behaviour(
    reason="a band path has no mesh to unfold, so it keeps no operations as before"
)
def test_band_path_without_table_keeps_no_operations():
    outcar = Outcar(EXAMPLES / "bands" / "non-colinear" / "OUTCAR")

    assert outcar.rotations.shape == (0, 3, 3)
