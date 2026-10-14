"""dosplot draws the projected DOS that VASP wrote, unscaled (issue #10).

Before the DOS rewrite, the projection modes scaled each PDOS curve by
total / sum(PDOS), so a Zr d curve of 0.61 states/eV in the DOSCAR was drawn
near 1.05. The reference here is a plain read of the DOSCAR, the other file VASP
wrote for the same run; the plotted curves come from vasprun.xml. The DOSCAR
prints 4 significant digits and vasprun.xml 4 decimals, per orbital.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from matplotlib.collections import PolyCollection

import pyprocar
from tests.utils import DATA_DIR

CALC = DATA_DIR / "examples" / "dos" / "non-spin-polarized"
FERMI = 5.3017
SPECIES_ATOMS = {"Sr": [0], "V": [1], "O": [2, 3, 4]}
SHELL_COLUMNS = {"s": [0], "p": [1, 2, 3], "d": [4, 5, 6, 7, 8]}
DOSCAR_RTOL = 5e-4
VASPRUN_ROUNDING = 5e-5


def read_doscar(path: Path) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list]:
    """Energies, total DOS and per-atom (energy, orbital) blocks of an ISPIN=1 DOSCAR."""
    lines = path.read_text().splitlines()
    natoms, nedos = int(lines[0].split()[0]), int(lines[5].split()[2])
    total = np.array([line.split() for line in lines[6 : 6 + nedos]], dtype=float)
    atoms = []
    for atom in range(natoms):
        start = 6 + nedos + atom * (nedos + 1) + 1
        rows = [line.split() for line in lines[start : start + nedos]]
        atoms.append(np.array(rows, dtype=float)[:, 1:])
    return total[:, 0], total[:, 1], atoms


def summed(atoms: list, atom_indices: list[int], columns: list[int]) -> npt.NDArray[np.float64]:
    return sum(atoms[a][:, columns].sum(axis=1) for a in atom_indices)


def assert_curve(y, expected, n_terms: int) -> None:
    np.testing.assert_allclose(
        y, expected, rtol=DOSCAR_RTOL, atol=n_terms * VASPRUN_ROUNDING + 1e-9
    )


def drawn_lines(ax) -> dict[str, npt.NDArray[np.float64]]:
    return {
        str(line.get_label()): np.asarray(line.get_ydata(), dtype=float)
        for line in ax.lines
        if not str(line.get_label()).startswith("_")
    }


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def doscar():
    if not CALC.exists():
        pytest.skip("fixture dos/non-spin-polarized not downloaded")
    return read_doscar(CALC / "DOSCAR")


GUARD = (
    "#10 reported PDOS curves drawn larger than the DOSCAR values; the DOS rewrite"
    " already plots unscaled sums, and this keeps them equal to the DOSCAR"
)


@pytest.mark.data
@pytest.mark.guards_existing_behaviour(reason=GUARD)
def test_overlay_orbitals_draws_the_doscar_shell_sums(doscar) -> None:
    energies, total, atoms = doscar

    _, ax = pyprocar.dosplot(
        code="vasp", dirname=str(CALC), mode="overlay_orbitals", fermi=FERMI, atoms=[1], show=False
    )

    curves = drawn_lines(ax)
    assert list(curves) == [
        "Total",
        r"$\mathrm{V}_{1}-(s)[\uparrow]$",
        r"$\mathrm{V}_{1}-(p)[\uparrow]$",
        r"$\mathrm{V}_{1}-(d)[\uparrow]$",
    ]
    np.testing.assert_allclose(ax.lines[0].get_xdata(), energies - FERMI, atol=5.5e-4)
    assert_curve(curves["Total"], total, 1)
    for shell, columns in SHELL_COLUMNS.items():
        assert_curve(
            curves[rf"$\mathrm{{V}}_{{1}}-({shell})[\uparrow]$"],
            summed(atoms, [1], columns),
            len(columns),
        )


@pytest.mark.data
@pytest.mark.guards_existing_behaviour(reason=GUARD)
def test_overlay_species_draws_the_doscar_species_sums(doscar) -> None:
    _, _, atoms = doscar

    _, ax = pyprocar.dosplot(
        code="vasp", dirname=str(CALC), mode="overlay_species", fermi=FERMI, show=False
    )

    curves = list(drawn_lines(ax).values())[1:]
    assert len(curves) == len(SPECIES_ATOMS)
    expected = {s: summed(atoms, a, list(range(9))) for s, a in SPECIES_ATOMS.items()}
    # The species curves follow the structure's species order, O, Sr, V for this POSCAR.
    for y, species in zip(curves, ["O", "Sr", "V"], strict=True):
        assert_curve(y, expected[species], 9 * len(SPECIES_ATOMS[species]))


@pytest.mark.data
@pytest.mark.guards_existing_behaviour(reason=GUARD)
def test_stack_orbitals_fills_up_to_the_cumulative_doscar_sums(doscar) -> None:
    energies, _, atoms = doscar

    _, ax = pyprocar.dosplot(
        code="vasp", dirname=str(CALC), mode="stack_orbitals", fermi=FERMI, atoms=[1], show=False
    )

    fills = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(fills) == len(SHELL_COLUMNS)
    cumulative = np.zeros_like(energies)
    n_terms = 0
    for fill, columns in zip(fills, SHELL_COLUMNS.values(), strict=True):
        cumulative = cumulative + summed(atoms, [1], columns)
        n_terms += len(columns)
        # fill_between's outline runs (x0, bottom0), (x0, top0) ... (xn, topn), (xn, bottomn) ...
        vertices = fill.get_paths()[0].vertices[1 : len(energies) + 1]
        np.testing.assert_allclose(vertices[:, 0], energies - FERMI, atol=5.5e-4)
        assert_curve(vertices[:, 1], cumulative, n_terms)
