import logging
from pathlib import Path

import numpy as np
import pytest

from pyprocar.core.property_store import Property
from pyprocar.io import vasp
from pyprocar.io.vasp.parser import VaspParser
from pyprocar.utils.log_utils import set_verbose_level

logger = logging.getLogger(__name__)

VERBOSE = 2
set_verbose_level(VERBOSE)


NON_COLINEAR_DOSCAR = """   2   2   1   0
  1.00000000  0.00000000  0.00000000  0.00000000  1.00000000
  0.00000000
  CAR 
 Test system                                
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.1200  0.0120
    -0.5000  0.2200  0.0220
     0.0000  0.3200  0.0320
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.0100  0.0200  0.0300  0.0400  0.0500  0.0600  0.0700  0.0800  0.0900  0.1000  0.1100  0.1200
    -0.5000  0.0110  0.0210  0.0310  0.0410  0.0510  0.0610  0.0710  0.0810  0.0910  0.1010  0.1110  0.1210
     0.0000  0.0120  0.0220  0.0320  0.0420  0.0520  0.0620  0.0720  0.0820  0.0920  0.1020  0.1120  0.1220
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.0130  0.0230  0.0330  0.0430  0.0530  0.0630  0.0730  0.0830  0.0930  0.1030  0.1130  0.1230
    -0.5000  0.0140  0.0240  0.0340  0.0440  0.0540  0.0640  0.0740  0.0840  0.0940  0.1040  0.1140  0.1240
     0.0000  0.0150  0.0250  0.0350  0.0450  0.0550  0.0650  0.0750  0.0850  0.0950  0.1050  0.1150  0.1250
"""


NON_SPIN_POLARIZED_DOSCAR = """   2   2   1   0
  1.00000000  0.00000000  0.00000000  0.00000000  1.00000000
  0.00000000
  CAR 
 Test system                                
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.1000  0.0100
    -0.5000  0.2000  0.0200
     0.0000  0.3000  0.0300
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.0100  0.0200  0.0300
    -0.5000  0.0400  0.0500  0.0600
     0.0000  0.0700  0.0800  0.0900
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.0150  0.0250  0.0350
    -0.5000  0.0450  0.0550  0.0650
     0.0000  0.0750  0.0850  0.0950
"""


SPIN_POLARIZED_DOSCAR = """   2   2   1   0
  1.00000000  0.00000000  0.00000000  0.00000000  1.00000000
  0.00000000
  CAR 
 Test system                                
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.1500 -0.1500  0.0150 -0.0150
    -0.5000  0.2500 -0.2500  0.0250 -0.0250
     0.0000  0.3500 -0.3500  0.0350 -0.0350
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.0100  0.0200  0.0300  0.0400
    -0.5000  0.0500  0.0600  0.0700  0.0800
     0.0000  0.0900  0.1000  0.1100  0.1200
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.0120  0.0220  0.0320  0.0420
    -0.5000  0.0520  0.0620  0.0720  0.0820
     0.0000  0.0920  0.1020  0.1120  0.1220
"""


TOTAL_DOS_ONLY = """   1   1   1   0
  1.00000000  0.00000000  0.00000000  0.00000000  1.00000000
  0.00000000
  CAR 
 Test system                                
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.1000  0.0100
    -0.5000  0.2000  0.0200
     0.0000  0.3000  0.0300
"""


@pytest.fixture
def non_spin_doscar_filepath(tmp_path: Path) -> Path:
    filepath = tmp_path / "DOSCAR_non_spin"
    filepath.write_text(NON_SPIN_POLARIZED_DOSCAR)
    return filepath


@pytest.fixture
def spin_doscar_filepath(tmp_path: Path) -> Path:
    filepath = tmp_path / "DOSCAR_spin"
    filepath.write_text(SPIN_POLARIZED_DOSCAR)
    return filepath


@pytest.fixture
def non_colinear_doscar_filepath(tmp_path: Path) -> Path:
    filepath = tmp_path / "DOSCAR_non_colinear"
    filepath.write_text(NON_COLINEAR_DOSCAR)
    return filepath


@pytest.fixture
def non_spin_doscar(non_spin_doscar_filepath: Path):
    return vasp.Doscar(non_spin_doscar_filepath)


@pytest.fixture
def spin_doscar(spin_doscar_filepath: Path):
    return vasp.Doscar(spin_doscar_filepath)


@pytest.fixture
def non_colinear_doscar(non_colinear_doscar_filepath: Path):
    return vasp.Doscar(non_colinear_doscar_filepath)


@pytest.fixture
def total_dos_only_filepath(tmp_path: Path) -> Path:
    filepath = tmp_path / "DOSCAR_total_only"
    filepath.write_text(TOTAL_DOS_ONLY)
    return filepath


@pytest.fixture
def total_dos_only_doscar(total_dos_only_filepath: Path):
    return vasp.Doscar(total_dos_only_filepath)


class TestDoscar:
    def test_doscar_from_str_sets_natoms(self) -> None:
        doscar = vasp.Doscar.from_str(SPIN_POLARIZED_DOSCAR)
        assert doscar.natoms == 2

    def test_doscar_is_spin_polarized(self, spin_doscar: vasp.Doscar) -> None:
        assert spin_doscar.is_spin_pol is True

    def test_doscar_is_not_spin_polarized(self, non_spin_doscar: vasp.Doscar) -> None:
        assert non_spin_doscar.is_spin_pol is False

    def test_doscar_energies_match_nedos(self, spin_doscar: vasp.Doscar) -> None:
        assert spin_doscar.energies.shape[0] == spin_doscar.nedos

    def test_doscar_total_dos_spin_shape(self, spin_doscar: vasp.Doscar) -> None:
        assert spin_doscar.total_dos.shape == (spin_doscar.nedos, 2)

    def test_doscar_total_dos_non_spin_shape(self, non_spin_doscar: vasp.Doscar) -> None:
        assert non_spin_doscar.total_dos.shape == (non_spin_doscar.nedos, 1)

    def test_doscar_integrated_dos_spin_shape(self, spin_doscar: vasp.Doscar) -> None:
        assert spin_doscar.integrated_dos.shape == (spin_doscar.nedos, 2)

    def test_doscar_integrated_dos_non_spin_shape(self, non_spin_doscar: vasp.Doscar) -> None:
        assert non_spin_doscar.integrated_dos.shape == (non_spin_doscar.nedos, 1)

    def test_doscar_projected_dos_spin_available(self, spin_doscar: vasp.Doscar) -> None:
        assert spin_doscar.projected_dos is not None

    def test_doscar_projected_dos_non_spin_available(self, non_spin_doscar: vasp.Doscar) -> None:
        assert non_spin_doscar.projected_dos is not None

    def test_doscar_projected_dos_non_colinear_available(
        self, non_colinear_doscar: vasp.Doscar
    ) -> None:
        pdos = non_colinear_doscar.projected_dos
        assert isinstance(pdos, np.ndarray)
        assert pdos is not None

    def test_doscar_projected_dos_spin_channels(self, spin_doscar: vasp.Doscar) -> None:
        pdos = spin_doscar.projected_dos
        assert isinstance(pdos, np.ndarray)
        assert pdos.shape[1] == 2

    def test_doscar_projected_dos_non_spin_channels(self, non_spin_doscar: vasp.Doscar) -> None:
        pdos = non_spin_doscar.projected_dos
        assert isinstance(pdos, np.ndarray)
        assert pdos.shape[1] == 1

    def test_doscar_projected_dos_non_colinear_channels(
        self, non_colinear_doscar: vasp.Doscar
    ) -> None:
        pdos = non_colinear_doscar.projected_dos
        assert isinstance(pdos, np.ndarray)
        assert pdos.shape[1] == 4

    def test_doscar_projected_dos_matches_atom_count(self, spin_doscar: vasp.Doscar) -> None:
        pdos = spin_doscar.projected_dos
        assert isinstance(pdos, np.ndarray)
        assert pdos.shape[2] == spin_doscar.natoms

    def test_doscar_projected_dos_matches_energy_count(self, non_spin_doscar: vasp.Doscar) -> None:
        pdos = non_spin_doscar.projected_dos
        assert isinstance(pdos, np.ndarray)
        assert pdos.shape[0] == non_spin_doscar.nedos

    def test_doscar_projected_dos_missing_returns_none(
        self, total_dos_only_doscar: vasp.Doscar
    ) -> None:
        assert total_dos_only_doscar.projected_dos is None


VASP_LM_NAMES = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2"]
VASP_F_NAMES = ["fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2", "fx3"]
COLLINEAR_OUTCAR = "   LNONCOLLINEAR =      F non collinear calculations\n E-fermi :   0.5000\n"
NON_COLLINEAR_OUTCAR = "   LNONCOLLINEAR =      T non collinear calculations\n E-fermi :   0.5000\n"
DOSCAR_NATOMS = 2
DOSCAR_NEDOS = 3


def _projected_value(atom: int, column: int, row: int) -> float:
    return atom + column / 100 + row / 10000


def _doscar_text(n_total: int, n_columns: int, n_atom_rows: int = DOSCAR_NEDOS) -> str:
    """A DOSCAR whose projected column c (1-based after the energy) of atom a, row r
    holds ``_projected_value(a, c, r)``. ``n_total`` is 1 (E, DOS, IntDOS) or 2
    (E, up, down, int up, int down) total DOS channels."""
    energy_line = f"    1.00000000    -1.00000000 {DOSCAR_NEDOS}      0.50000000      1.00000000"
    lines = [
        f"   {DOSCAR_NATOMS}   {DOSCAR_NATOMS}   1   0",
        "  1.00000000  0.00000000  0.00000000  0.00000000  1.00000000",
        "  0.00000000",
        "  CAR ",
        " Test system",
        energy_line,
    ]
    energies = [-1.0 + 0.5 * row for row in range(DOSCAR_NEDOS)]
    for row, energy in enumerate(energies):
        values = [0.1 * (row + 1) + 0.01 * channel for channel in range(2 * n_total)]
        lines.append(f"{energy:10.4f}" + "".join(f"{v:10.4f}" for v in values))
    for atom in range(DOSCAR_NATOMS):
        lines.append(energy_line)
        for row, energy in enumerate(energies[:n_atom_rows]):
            values = [_projected_value(atom, c, row) for c in range(1, n_columns + 1)]
            lines.append(f"{energy:10.4f}" + "".join(f"{v:10.4f}" for v in values))
    return "\n".join(lines) + "\n"


def _dos_from(doscar: str, outcar: str | None = None):
    dos = VaspParser.from_str(doscar=doscar, outcar=outcar).dos
    assert dos is not None
    return dos


class TestDosFromDoscarOnly:
    """A VASP directory without vasprun.xml takes its DOS from the DOSCAR (issue #10)."""

    @pytest.mark.parametrize(
        ("n_total", "n_columns", "outcar", "n_spins", "orbital_names"),
        [
            pytest.param(1, 3, None, 1, ["s", "p", "d"], id="LORBIT=10"),
            pytest.param(1, 4, None, 1, ["s", "p", "d", "f"], id="LORBIT=10-f"),
            pytest.param(1, 9, None, 1, VASP_LM_NAMES, id="LORBIT=11"),
            pytest.param(1, 16, COLLINEAR_OUTCAR, 1, VASP_LM_NAMES + VASP_F_NAMES, id="LORBIT=11-f"),
            pytest.param(2, 6, None, 2, ["s", "p", "d"], id="ISPIN=2-LORBIT=10"),
            pytest.param(2, 18, None, 2, VASP_LM_NAMES, id="ISPIN=2"),
            pytest.param(2, 32, None, 2, VASP_LM_NAMES + VASP_F_NAMES, id="ISPIN=2-f"),
            pytest.param(1, 12, None, 4, ["s", "p", "d"], id="non-collinear-LORBIT=10"),
            pytest.param(1, 36, None, 4, VASP_LM_NAMES, id="non-collinear"),
            pytest.param(1, 16, NON_COLLINEAR_OUTCAR, 4, ["s", "p", "d", "f"], id="non-collinear-spdf"),
            pytest.param(1, 64, None, 4, VASP_LM_NAMES + VASP_F_NAMES, id="non-collinear-f"),
        ],
    )
    def test_projected_dos_follows_vasps_column_order(
        self, n_total, n_columns, outcar, n_spins, orbital_names
    ) -> None:
        # VASP writes each orbital's spin columns together: s_up s_down py_up ... for
        # ISPIN=2 and s_tot s_mx s_my s_mz py_tot ... for non-collinear runs.
        dos = _dos_from(_doscar_text(n_total, n_columns), outcar)

        expected = [
            [
                [
                    [_projected_value(a, 1 + o * n_spins + s, r) for o in range(len(orbital_names))]
                    for a in range(DOSCAR_NATOMS)
                ]
                for s in range(n_spins)
            ]
            for r in range(DOSCAR_NEDOS)
        ]
        assert dos.projected is not None
        np.testing.assert_allclose(dos.projected.value, expected)
        assert dos.orbital_names == orbital_names
        assert dos.n_spin_channels == n_total

    def test_total_dos_and_fermi_come_from_the_doscar(self) -> None:
        dos = _dos_from(TOTAL_DOS_ONLY)

        np.testing.assert_allclose(dos.energies, [-1.0, -0.5, 0.0])
        np.testing.assert_allclose(dos.total.value, [[0.1], [0.2], [0.3]])
        assert dos.fermi == 0.5
        assert dos.projected is None

    def test_sixteen_columns_without_an_outcar_raise(self) -> None:
        # 16 columns are the lm orbitals with f of a collinear run or s, p, d, f
        # times (total, mx, my, mz) of a non-collinear one; only the OUTCAR tells.
        with pytest.raises(ValueError, match="LNONCOLLINEAR"):
            _ = VaspParser.from_str(doscar=_doscar_text(1, 16)).dos

    @pytest.mark.parametrize(
        ("n_total", "n_columns", "match"),
        [(1, 5, "5 projected columns"), (2, 7, "7 projected columns"), (2, 10, "5 orbitals")],
    )
    def test_unknown_column_count_raises(self, n_total, n_columns, match) -> None:
        with pytest.raises(ValueError, match=match):
            _ = VaspParser.from_str(doscar=_doscar_text(n_total, n_columns)).dos

    def test_truncated_projected_block_raises(self) -> None:
        with pytest.raises(ValueError, match="DOSCAR"):
            _ = VaspParser.from_str(doscar=_doscar_text(1, 9, n_atom_rows=2)).dos

    def test_atoms_select_without_a_structure_and_species_raise(self) -> None:
        dos = _dos_from(_doscar_text(1, 9))

        assert dos.structure is None
        summed = dos.compute_projected_sum(atoms=[1], orbitals=[4, 5, 6, 7, 8])
        assert isinstance(summed, Property)
        np.testing.assert_allclose(
            summed.value[:, 0], [sum(_projected_value(1, c, r) for c in range(5, 10)) for r in range(3)]
        )
        with pytest.raises(ValueError, match="structure"):
            dos.compute_projected_sum(species=["Zr"])
