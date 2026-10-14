"""A VASP directory without vasprun.xml reads its DOS from the DOSCAR (issue #10).

VASP wrote both files in each fixture, so the vasprun.xml DOS of the same run is
an independent reference. The DOSCAR prints energies to 3 decimals and DOS values
to 4 significant digits; vasprun.xml prints both to 4 decimals.
"""

from pathlib import Path

import numpy as np
import pytest

from pyprocar.core.dos import DensityOfStates
from pyprocar.core.property_store import Property
from tests.utils import DATA_DIR, writable_copy

DOS_DIR = DATA_DIR / "examples" / "dos"
ENERGY_ATOL = 5e-4 + 5e-5
DOS_RTOL = 5e-4
DOS_ATOL = 5e-5


def _doscar_only(tmp_path: Path, name: str, keep: tuple[str, ...]) -> Path:
    src = DOS_DIR / name
    if not src.exists():
        pytest.skip(f"fixture dos/{name} not downloaded")

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {n for n in names if n not in keep}

    return writable_copy(src, tmp_path / name, ignore=ignore)


@pytest.mark.data
@pytest.mark.parametrize("name", ["non-spin-polarized", "spin-polarized", "non-colinear"])
def test_doscar_dos_equals_the_vasprun_dos_of_the_same_run(tmp_path: Path, name: str) -> None:
    calc = _doscar_only(tmp_path, name, keep=("DOSCAR", "OUTCAR", "POSCAR"))
    reference = DensityOfStates.from_code("vasp", str(DOS_DIR / name))

    dos = DensityOfStates.from_code("vasp", str(calc))

    np.testing.assert_allclose(dos.energies, reference.energies, rtol=0, atol=ENERGY_ATOL)
    # The DOSCAR of a non-collinear run holds the total DOS only, not its Sx, Sy, Sz.
    n_total = dos.n_spin_channels
    assert n_total == (1 if name == "non-colinear" else reference.n_spin_channels)
    np.testing.assert_allclose(
        dos.total.value, reference.total.value[:, :n_total], rtol=DOS_RTOL, atol=DOS_ATOL
    )
    assert dos.projected is not None and reference.projected is not None
    np.testing.assert_allclose(
        dos.projected.value, reference.projected.value, rtol=DOS_RTOL, atol=DOS_ATOL
    )
    assert dos.orbital_names == reference.orbital_names
    assert dos.fermi == reference.fermi
    assert dos.species == reference.species


@pytest.mark.data
def test_doscar_dos_without_a_poscar_selects_atoms_and_rejects_species(tmp_path: Path) -> None:
    # The reporter's directory held DOSCAR, OUTCAR, PROCAR and EIGENVAL only.
    calc = _doscar_only(tmp_path, "non-spin-polarized", keep=("DOSCAR", "OUTCAR"))
    reference = DensityOfStates.from_code("vasp", str(DOS_DIR / "non-spin-polarized"))
    selection = {"atoms": [1], "orbitals": [4, 5, 6, 7, 8]}

    dos = DensityOfStates.from_code("vasp", str(calc))

    assert dos.structure is None
    summed, expected = (
        dos.compute_projected_sum(**selection),
        reference.compute_projected_sum(**selection),
    )
    assert isinstance(summed, Property) and isinstance(expected, Property)
    np.testing.assert_allclose(
        summed.value,
        expected.value,
        rtol=DOS_RTOL,
        atol=5 * DOS_ATOL,
    )
    with pytest.raises(ValueError, match="no structure"):
        dos.compute_projected_sum(species=["V"])
