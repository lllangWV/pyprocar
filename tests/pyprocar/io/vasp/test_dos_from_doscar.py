"""A VASP directory without vasprun.xml reads its DOS from the DOSCAR (issue #10).

VASP wrote both files in each fixture, so the vasprun.xml DOS of the same run is
an independent reference. The DOSCAR prints energies to 3 decimals and DOS values
to 4 significant digits; vasprun.xml prints both to 4 decimals.

A DOS comes from one source: the <dos> block of vasprun.xml for the run's k-mesh,
else the DOSCAR. Its energies, total and projected DOS and its Fermi energy all come
from that source; the OUTCAR's E-fermi never feeds it (#75).
"""

import re
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from pyprocar.core.dos import DensityOfStates
from pyprocar.core.property_store import Property
from pyprocar.io.vasp.outcar import Outcar
from pyprocar.io.vasp.vasprun import VaspXML
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


def _efermi(block: ElementTree.Element) -> float:
    """The efermi VASP writes in a <dos> block."""
    efermi = block.findtext("i[@name='efermi']")
    assert efermi is not None
    return float(efermi)


NON_SPIN_POLARIZED_EFERMI = '   <i name="efermi">      5.69903164 </i>\n'


def _without_efermi(vasprun: Path) -> None:
    _ = vasprun.write_text(_edited(vasprun, NON_SPIN_POLARIZED_EFERMI, ""))


@pytest.mark.data
@pytest.mark.parametrize("name", ["non-spin-polarized", "spin-polarized", "non-colinear"])
def test_a_vasprun_without_outcar_or_doscar_gives_its_own_fermi_energy(
    tmp_path: Path, name: str
) -> None:
    calc = _doscar_only(tmp_path, name, keep=("POSCAR", "vasprun.xml"))
    efermi = _efermi(_dos_blocks(calc / "vasprun.xml")[None])
    # VASP writes the same value in the DOSCAR header of the run, 4th field of line 6.
    doscar_header = (DOS_DIR / name / "DOSCAR").read_text().splitlines()[5].split()
    assert efermi == float(doscar_header[3])

    dos = DensityOfStates.from_code("vasp", str(calc))

    assert dos.fermi == efermi


@pytest.mark.data
def test_a_vasprun_without_efermi_borrows_no_fermi_energy_from_the_outcar_or_doscar(
    tmp_path: Path,
) -> None:
    # The OUTCAR (E-fermi 5.6990) and DOSCAR (header 5.69903164) of the same run both
    # hold a Fermi energy, but the DOS comes from vasprun.xml, whose <dos> block lost it.
    calc = _doscar_only(
        tmp_path, "non-spin-polarized", keep=("DOSCAR", "OUTCAR", "POSCAR", "vasprun.xml")
    )
    _without_efermi(calc / "vasprun.xml")

    with pytest.raises(ValueError, match=re.escape(str(calc / "vasprun.xml")) + ".*efermi"):
        _ = DensityOfStates.from_code("vasp", str(calc))


def _dos_blocks(vasprun: Path) -> dict[str | None, ElementTree.Element]:
    """The <dos> blocks of a vasprun.xml by their comment, read with the standard library."""
    return {block.get("comment"): block for block in ElementTree.parse(vasprun).iter("dos")}


def _rows(spin_sets: list[ElementTree.Element]) -> np.ndarray:
    """(energy, spin, column) from the <set> of each spin; column 0 is the energy."""
    return np.stack(
        [
            np.array([[float(x) for x in (r.text or "").split()] for r in s.iter("r")])
            for s in spin_sets
        ],
        axis=1,
    )


def _block_dos(block: ElementTree.Element) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Energies, total (energy, spin), projected (energy, spin, ion, orbital) and efermi."""
    total = _rows(block.findall("total/array/set/set"))
    ions = block.findall("partial/array/set/set")
    projected = np.stack([_rows(ion.findall("set"))[..., 1:] for ion in ions], axis=2)
    return total[:, 0, 0], total[..., 1], projected, _efermi(block)


def _doubled_rows(block: str) -> str:
    """The block with every DOS value doubled; the energy column stays."""

    def double(row: re.Match[str]) -> str:
        energy, *values = row[1].split()
        return "<r>" + " ".join([energy, *(f"{2 * float(v):.4f}" for v in values)]) + "</r>"

    return re.sub(r"<r>(.*?)</r>", double, block)


@pytest.mark.data
def test_a_kpoints_opt_dos_block_written_first_is_not_the_dos(tmp_path: Path) -> None:
    # VASP 6 with KPOINTS_OPT writes a <dos comment="kpoints_opt"> block for the extra
    # k-points before the <dos> block of the run's own k-mesh, in the same calculation.
    calc = _doscar_only(tmp_path, "non-spin-polarized", keep=("POSCAR", "vasprun.xml"))
    vasprun = calc / "vasprun.xml"
    text = vasprun.read_text()
    start = text.index("  <dos>")
    block = text[start : text.index("</dos>") + len("</dos>")]
    assert block.count(NON_SPIN_POLARIZED_EFERMI) == 1
    opt_block = _doubled_rows(
        block.replace("<dos>", '<dos comment="kpoints_opt">').replace(
            NON_SPIN_POLARIZED_EFERMI, '   <i name="efermi">      4.20000000 </i>\n'
        )
    )
    _ = vasprun.write_text(_edited(vasprun, "  <dos>\n", opt_block + "\n  <dos>\n"))
    blocks = _dos_blocks(vasprun)
    energies, total, projected, efermi = _block_dos(blocks[None])
    _, opt_total, opt_projected, opt_efermi = _block_dos(blocks["kpoints_opt"])
    assert (efermi, opt_efermi) == (5.69903164, 4.2)
    assert not np.allclose(opt_total, total) and not np.allclose(opt_projected, projected)

    dos = DensityOfStates.from_code("vasp", str(calc))

    np.testing.assert_array_equal(dos.energies, energies)
    np.testing.assert_array_equal(dos.total.value, total)
    assert dos.projected is not None
    np.testing.assert_array_equal(dos.projected.value, projected)
    assert dos.fermi == efermi


def _add_other_run_file(calc: Path, name: str) -> None:
    """Put the spin-polarized run's ``name`` into ``calc``. Its DOSCAR header and
    vasprun.xml give the Fermi energy 4.96110661, its OUTCAR 4.9611."""
    src = DOS_DIR / "spin-polarized"
    if not src.exists():
        pytest.skip("fixture dos/spin-polarized not downloaded")
    _ = writable_copy(src / name, calc / name)


@pytest.mark.data
def test_a_vasprun_next_to_another_runs_doscar_gives_its_own_dos_and_fermi_energy(
    tmp_path: Path,
) -> None:
    calc = _doscar_only(tmp_path, "non-spin-polarized", keep=("POSCAR", "vasprun.xml"))
    _add_other_run_file(calc, "DOSCAR")
    energies, total, projected, efermi = _block_dos(_dos_blocks(calc / "vasprun.xml")[None])
    assert efermi == 5.69903164

    dos = DensityOfStates.from_code("vasp", str(calc))

    np.testing.assert_array_equal(dos.energies, energies)
    np.testing.assert_array_equal(dos.total.value, total)
    assert dos.projected is not None
    np.testing.assert_array_equal(dos.projected.value, projected)
    assert dos.fermi == efermi


@pytest.mark.data
def test_a_doscar_next_to_another_runs_outcar_gives_its_own_fermi_energy(
    tmp_path: Path,
) -> None:
    calc = _doscar_only(tmp_path, "non-spin-polarized", keep=("DOSCAR", "POSCAR"))
    _add_other_run_file(calc, "OUTCAR")

    dos = DensityOfStates.from_code("vasp", str(calc))

    # The DOSCAR header of this run: "12.62318812 -31.06108337 6000 5.69903164 1.0".
    assert dos.fermi == 5.69903164


def _large_vasprun() -> VaspXML:
    calc = DATA_DIR / "examples" / "fermi2d" / "bisb_monolayer"
    if not calc.exists():
        pytest.skip("fixture fermi2d/bisb_monolayer not downloaded")
    return VaspXML(calc / "vasprun.xml")


@pytest.mark.data
@pytest.mark.guards_existing_behaviour(
    reason="main finds this DOS block; the one-block lookup must still find it in a 281 MB file"
)
def test_the_dos_block_of_a_large_vasprun_is_found() -> None:
    vaspxml = _large_vasprun()

    assert vaspxml.has_dos
    assert vaspxml.dos_energies is not None
    assert vaspxml.dos_energies.shape == (301,)  # '<i type="int" name="NEDOS">   301</i>'


@pytest.mark.data
def test_the_fermi_energy_of_a_large_vasprun_comes_from_its_dos_block() -> None:
    # Fails on main, whose VaspXML has no fermi (#75).
    vaspxml = _large_vasprun()

    assert vaspxml.fermi == -1.21843928  # '<i name="efermi">     -1.21843928 </i>' in the file
