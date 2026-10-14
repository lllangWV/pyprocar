"""Species modes need the structure; without a POSCAR or vasprun.xml they say so (issue #10).

The reporter's directory held DOSCAR, OUTCAR, PROCAR and EIGENVAL only, so the
density of states and the band structure carry no structure and no species.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import pyprocar
from tests.utils import DATA_DIR, writable_copy

NEEDS_STRUCTURE = r"(?s)species.*structure.*POSCAR.*atoms="


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _without_structure(tmp_path: Path, relpath: str, keep: tuple[str, ...]) -> str:
    src = DATA_DIR / "examples" / relpath
    if not src.exists():
        pytest.skip(f"fixture {relpath} not downloaded")

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {n for n in names if n not in keep}

    return str(writable_copy(src, tmp_path / "calc", ignore=ignore))


@pytest.mark.data
@pytest.mark.parametrize("mode", ["overlay_species", "stack_species"])
def test_dosplot_species_modes_without_a_structure_raise(tmp_path: Path, mode: str) -> None:
    calc = _without_structure(tmp_path, "dos/non-spin-polarized", keep=("DOSCAR", "OUTCAR"))

    with pytest.raises(ValueError, match=NEEDS_STRUCTURE):
        pyprocar.dosplot(code="vasp", dirname=calc, mode=mode, show=False)


@pytest.mark.data
def test_bandsplot_species_mode_without_a_structure_raises(tmp_path: Path) -> None:
    calc = _without_structure(
        tmp_path, "bands/non-spin-polarized", keep=("OUTCAR", "PROCAR", "KPOINTS")
    )

    with pytest.raises(ValueError, match=NEEDS_STRUCTURE):
        pyprocar.bandsplot(code="vasp", dirname=calc, mode="overlay_species", show=False)
