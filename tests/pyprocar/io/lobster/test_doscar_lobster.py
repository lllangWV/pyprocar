"""Tests for DoscarLobster extractor."""

import numpy as np
import pytest

from pyprocar.io.lobster import DoscarLobster

# Minimal DOSCAR.lobster content (non-spin-polarized)
DOSCAR_CONTENT = """   2   2   0   1
  10.0000   3.0000   3.0000   3.0000   0.0010
  0.00000
 CAR
 Fe2O3
 -10.0000  10.0000   5   0.0000   1.0000
 -10.0000   0.1000   0.0500
  -5.0000   0.5000   0.3000
   0.0000   1.0000   0.8000
   5.0000   0.5000   0.9500
  10.0000   0.1000   1.0000
"""


class TestDoscarLobster:
    def test_from_str(self):
        extractor = DoscarLobster.from_str(DOSCAR_CONTENT)
        assert extractor is not None

    def test_nedos(self):
        extractor = DoscarLobster.from_str(DOSCAR_CONTENT)
        assert extractor.nedos == 5

    def test_energies(self):
        extractor = DoscarLobster.from_str(DOSCAR_CONTENT)
        assert len(extractor.energies) == 5
        assert extractor.energies[0] == pytest.approx(-10.0)
        assert extractor.energies[-1] == pytest.approx(10.0)

    def test_total_dos_shape(self):
        extractor = DoscarLobster.from_str(DOSCAR_CONTENT)
        assert extractor.total_dos.shape == (5, 1)  # (nedos, n_spins)

    def test_is_spin_polarized(self):
        extractor = DoscarLobster.from_str(DOSCAR_CONTENT)
        assert not extractor.is_spin_polarized

    def test_mapping_protocol(self):
        extractor = DoscarLobster.from_str(DOSCAR_CONTENT)
        assert "energies" in extractor
        assert len(extractor) == 5


# The preamble and header lines LOBSTER 5.1.1 writes (QE diamond, nspin = 2), with NEDOS cut to 3.
# Every basis-function label carries its principal quantum number.
LOBSTER_5_PREAMBLE = """   2    2    1    2
  1.1373193e+01   2.5242506e-10   2.5242506e-10   2.5242506e-10   0.0000000e+00
            0.000000e+00
  CAR
 DOS and projected DOS file by LOBSTER."""
LOBSTER_5_PREAMBLE += " Energy is shifted such that the Fermi level lies at 0 eV. \n"
LOBSTER_5_TOTAL_HEADER = "     33.49182335      -6.55823923     3      15.46929519       1.00000000"
LOBSTER_5_ATOM_HEADER = (
    "     18.02252816     -22.02753442     3      15.46929519       1.00000000; Z=  6; "
)
ENERGIES = (-1.0, 0.0, 1.0)


def _doscar(labels_per_atom: list[str], columns_per_label: int) -> str:
    """A DOSCAR.lobster whose atom blocks hold 10 * atom + c + i / 10 in column c of row i."""
    n_total = 2 * columns_per_label
    lines = [LOBSTER_5_TOTAL_HEADER]
    lines += [f"{e:10.5f}" + "   1.00000" * n_total for e in ENERGIES]
    for iatom, labels in enumerate(labels_per_atom):
        lines.append(LOBSTER_5_ATOM_HEADER + labels + " ")
        n_columns = len(labels.split()) * columns_per_label
        lines += [
            f"{e:10.5f}"
            + "".join(f" {10 * iatom + c + i / 10:9.5f}" for c in range(1, n_columns + 1))
            for i, e in enumerate(ENERGIES)
        ]
    return LOBSTER_5_PREAMBLE + "\n".join(lines) + "\n"


def test_lobster_5_labels_with_principal_numbers_fill_their_spin_and_orbital():
    doscar = DoscarLobster.from_str(_doscar(["2s 2p_y 2p_z 2p_x"], columns_per_label=2))

    projected = doscar.projected_dos
    assert projected is not None
    rows = np.array([0.0, 0.1, 0.2])
    # spin-polarized columns alternate up and down for each label: 2s up, 2s down, 2p_y up, ...
    for column, orbital in enumerate(["s", "p_y", "p_z", "p_x"]):
        iorb = doscar.orbital_labels.index(orbital)
        np.testing.assert_allclose(projected[:, 0, 0, iorb], 2 * column + 1 + rows)
        np.testing.assert_allclose(projected[:, 1, 0, iorb], 2 * column + 2 + rows)
    assert np.all(projected[:, :, :, 4:] == 0.0)


def test_shells_that_share_an_orbital_add_into_its_column():
    doscar = DoscarLobster.from_str(_doscar(["4s 5s 4p_y 4p_z 4p_x"], columns_per_label=1))

    projected = doscar.projected_dos
    assert projected is not None
    s = projected[:, 0, 0, doscar.orbital_labels.index("s")]
    np.testing.assert_allclose(s, [1.0 + 2.0, 1.1 + 2.1, 1.2 + 2.2])


def test_d_and_f_labels_fill_the_core_slots_they_name():
    f_labels = "4f_y(3x^2-y^2) 4f_xyz 4f_yz^2 4f_z^3 4f_xz^2 4f_z(x^2-y^2) 4f_x(x^2-3y^2)"
    doscar = DoscarLobster.from_str(
        _doscar(["3d_xy 3d_yz 3d_z^2 3d_xz 3d_x^2-y^2", f"6s {f_labels}"], columns_per_label=1)
    )

    projected = doscar.projected_dos
    assert projected is not None
    assert projected.shape == (3, 1, 2, 16)
    assert len(doscar.orbital_labels) == 16
    # in the core's orbital order, slots 4 to 15 hold dxy, dyz, dz2, dxz, dx2-y2, then f m = -3..3
    np.testing.assert_allclose(projected[0, 0, 0, 4:9], [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(projected[0, 0, 1, 9:16], [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
    assert doscar.orbital_labels[10] == "f_xyz"


@pytest.mark.guards_existing_behaviour(
    reason="adding the f slots must not grow the orbital axis of an s, p, d basis"
)
def test_a_basis_without_f_keeps_nine_orbitals():
    doscar = DoscarLobster.from_str(_doscar(["3d_xy 4s"], columns_per_label=1))

    projected = doscar.projected_dos
    assert projected is not None and projected.shape == (3, 1, 1, 9)
    assert len(doscar.orbital_labels) == 9


def test_an_unknown_orbital_label_raises_naming_it():
    doscar = DoscarLobster.from_str(_doscar(["2s 2q_x"], columns_per_label=1))

    with pytest.raises(ValueError, match=r"'2q_x'"):
        _ = doscar.projected_dos
