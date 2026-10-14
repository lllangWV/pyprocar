"""A VASP directory without vasprun.xml reads its DOS from the DOSCAR (issue #10).

VASP wrote both files in each fixture, so the vasprun.xml DOS of the same run is
an independent reference. The DOSCAR prints energies to 3 decimals and DOS values
to 4 significant digits; vasprun.xml prints both to 4 decimals.

The Fermi energy comes from the OUTCAR, else the DOSCAR header, else vasprun.xml (#75).
"""

import re
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from pyprocar.core.dos import DensityOfStates
from pyprocar.core.property_store import Property
from pyprocar.io.vasp.outcar import Outcar
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
    d_orbitals = [4, 5, 6, 7, 8]

    dos = DensityOfStates.from_code("vasp", str(calc))

    assert dos.structure is None
    summed, expected = (
        dos.compute_projected_sum(atoms=[1], orbitals=d_orbitals),
        reference.compute_projected_sum(atoms=[1], orbitals=d_orbitals),
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


# The OUTCAR echoes the INCAR verbatim at its top and lists the parameters VASP ran
# with after "Startparameter for this run:". LSORBIT turns LNONCOLLINEAR on.
NON_COLINEAR_ECHO = "   LNONCOLLINEAR = .TRUE.\n"


def _listed(value: str) -> str:
    return f"   LNONCOLLINEAR = {value:>6} non collinear calculations\n"


NON_COLINEAR_LISTED = _listed("T")


def _edited(path: Path, old: str, new: str) -> str:
    text = path.read_text()
    assert text.count(old) == 1, f"{old!r} occurs {text.count(old)} times in {path}"
    return text.replace(old, new)


def _fixture_outcar(name: str) -> Path:
    path = DOS_DIR / name / "OUTCAR"
    if not path.exists():
        pytest.skip(f"fixture dos/{name} not downloaded")
    return path


@pytest.mark.data
@pytest.mark.parametrize(
    ("name", "old", "new", "expected"),
    [
        pytest.param("non-colinear", "", "", True, id="non-collinear"),
        pytest.param("non-spin-polarized", "", "", False, id="collinear"),
        pytest.param(
            "non-colinear", NON_COLINEAR_ECHO, "   LNONCOLLINEAR = F\n", True, id="echo-says-F"
        ),
        pytest.param(
            "non-colinear",
            NON_COLINEAR_LISTED,
            _listed("F"),
            True,
            id="LSORBIT-listed-T",
        ),
        pytest.param(
            "non-colinear",
            NON_COLINEAR_LISTED,
            _listed(".TRUE."),
            True,
            id="listed-as-.TRUE.",
        ),
        pytest.param(
            "non-spin-polarized",
            _listed("F"),
            _listed(".FALSE."),
            False,
            id="listed-as-.FALSE.",
        ),
    ],
)
def test_outcar_non_collinear_is_the_value_vasp_ran_with(
    name: str, old: str, new: str, expected: bool
) -> None:
    path = _fixture_outcar(name)
    text = _edited(path, old, new) if old else path.read_text()

    assert Outcar.from_str(text).is_non_collinear is expected


@pytest.mark.data
def test_outcar_without_its_parameter_listing_gives_no_non_collinear() -> None:
    # A run killed before VASP listed its parameters leaves only the INCAR echo.
    text = _edited(_fixture_outcar("non-colinear"), NON_COLINEAR_ECHO, "   LNONCOLLINEAR = T\n")
    killed = text[: text.index(" Startparameter for this run:")]

    assert Outcar.from_str(killed).is_non_collinear is None


def _as_spdf(doscar: Path) -> None:
    """Rewrite the projected blocks of a LORBIT=11 non-collinear DOSCAR (9 lm orbitals
    times total, mx, my, mz) as s, p, d, f times the same 4, with f zero: 16 columns."""
    lines = doscar.read_text().splitlines()
    natoms, nedos = int(lines[0].split()[0]), int(lines[5].split()[2])
    for atom in range(natoms):
        first = 6 + nedos + 1 + atom * (nedos + 1)
        for i in range(first, first + nedos):
            row = [float(v) for v in lines[i].split()]
            lm = np.array(row[1:]).reshape(9, 4)
            spdf = [lm[0], lm[1:4].sum(axis=0), lm[4:9].sum(axis=0), np.zeros(4)]
            lines[i] = f"{row[0]:10.3f}" + "".join(f"{v:12.4E}" for v in np.ravel(spdf))
    doscar.write_text("\n".join(lines) + "\n")


@pytest.mark.data
def test_sixteen_columns_follow_the_run_parameters_not_the_incar_echo(tmp_path: Path) -> None:
    calc = _doscar_only(tmp_path, "non-colinear", keep=("DOSCAR", "OUTCAR", "POSCAR"))
    outcar = calc / "OUTCAR"
    _ = outcar.write_text(_edited(outcar, NON_COLINEAR_ECHO, "   LNONCOLLINEAR = F\n"))
    _as_spdf(calc / "DOSCAR")
    reference = DensityOfStates.from_code("vasp", str(DOS_DIR / "non-colinear"))

    dos = DensityOfStates.from_code("vasp", str(calc))

    assert dos.orbital_names == ["s", "p", "d", "f"]
    assert dos.projected is not None and reference.projected is not None
    np.testing.assert_allclose(
        dos.projected.value[..., 2],
        reference.projected.value[..., 4:9].sum(axis=-1),
        rtol=DOS_RTOL,
        atol=5 * DOS_ATOL,
    )


def _cut_before_e_fermi(outcar: Path) -> Path:
    """Cut the OUTCAR where an interrupted run stops: before VASP writes "E-fermi :"."""
    text = outcar.read_text()
    _ = outcar.write_text(text[: text.index(" E-fermi :")])
    return outcar


@pytest.mark.data
def test_an_outcar_cut_before_e_fermi_leaves_the_fermi_energy_to_the_doscar(
    tmp_path: Path,
) -> None:
    calc = _doscar_only(
        tmp_path, "non-spin-polarized", keep=("DOSCAR", "OUTCAR", "POSCAR", "vasprun.xml")
    )
    _ = _cut_before_e_fermi(calc / "OUTCAR")

    dos = DensityOfStates.from_code("vasp", str(calc))

    # The DOSCAR header of this run: "12.62318812 -31.06108337 6000 5.69903164 1.0".
    assert dos.fermi == 5.69903164


def _vasprun_efermi(vasprun: Path) -> float:
    """The efermi VASP writes in the <dos> block, read with the standard library's XML parser."""
    (dos,) = ElementTree.parse(vasprun).getroot().iter("dos")
    efermi = dos.find("i[@name='efermi']")
    assert efermi is not None and efermi.text is not None
    return float(efermi.text)


def _without_efermi(vasprun: Path) -> Path:
    text = vasprun.read_text()
    edited = re.sub(r'\s*<i name="efermi">[^<]*</i>', "", text)
    assert edited != text
    _ = vasprun.write_text(edited)
    return vasprun


@pytest.mark.data
@pytest.mark.parametrize("name", ["non-spin-polarized", "spin-polarized", "non-colinear"])
def test_a_vasprun_without_outcar_or_doscar_gives_its_own_fermi_energy(
    tmp_path: Path, name: str
) -> None:
    calc = _doscar_only(tmp_path, name, keep=("POSCAR", "vasprun.xml"))
    efermi = _vasprun_efermi(calc / "vasprun.xml")
    # VASP writes the same value in the DOSCAR header of the run, 4th field of line 6.
    doscar_header = (DOS_DIR / name / "DOSCAR").read_text().splitlines()[5].split()
    assert efermi == float(doscar_header[3])

    dos = DensityOfStates.from_code("vasp", str(calc))

    assert dos.fermi == efermi


@pytest.mark.data
def test_a_vasprun_without_efermi_and_no_other_source_raises_naming_the_directory(
    tmp_path: Path,
) -> None:
    calc = _doscar_only(tmp_path, "non-spin-polarized", keep=("POSCAR", "vasprun.xml"))
    _ = _without_efermi(calc / "vasprun.xml")

    with pytest.raises(ValueError, match="Fermi energy in " + re.escape(str(calc))):
        _ = DensityOfStates.from_code("vasp", str(calc))


@pytest.mark.data
def test_an_outcar_cut_before_e_fermi_leaves_the_fermi_energy_to_the_vasprun(
    tmp_path: Path,
) -> None:
    calc = _doscar_only(tmp_path, "non-spin-polarized", keep=("OUTCAR", "POSCAR", "vasprun.xml"))
    _ = _cut_before_e_fermi(calc / "OUTCAR")

    dos = DensityOfStates.from_code("vasp", str(calc))

    # The vasprun.xml of this run: <i name="efermi">      5.69903164 </i>.
    assert dos.fermi == 5.69903164


@pytest.mark.data
def test_an_outcar_cut_before_e_fermi_with_no_other_source_raises_naming_it(
    tmp_path: Path,
) -> None:
    calc = _doscar_only(tmp_path, "non-spin-polarized", keep=("OUTCAR", "POSCAR", "vasprun.xml"))
    outcar = _cut_before_e_fermi(calc / "OUTCAR")
    _ = _without_efermi(calc / "vasprun.xml")

    with pytest.raises(ValueError, match=re.escape(str(outcar)) + ".*E-fermi"):
        _ = DensityOfStates.from_code("vasp", str(calc))
