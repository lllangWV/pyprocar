import numpy as np
import pytest

from pyprocar.io.lobster.doscar_lobster import LOBSTER_ORBITALS
from pyprocar.io.lobster.parser import LobsterParser
from tests.pyprocar.io.lobster.test_doscar_lobster import DOSCAR_CONTENT
from tests.pyprocar.io.lobster.test_fatband import FATBAND_CONTENT
from tests.pyprocar.io.lobster.test_lobsterout import LOBSTEROUT_CONTENT


def test_fatband_weights_land_on_their_spin_ion_and_orbital(tmp_path):
    for name, content in {
        "lobsterout": LOBSTEROUT_CONTENT,
        "DOSCAR.lobster": DOSCAR_CONTENT,
        "FATBAND_Fe_s.lobster": FATBAND_CONTENT,
    }.items():
        (tmp_path / name).write_text(content)

    ebs = LobsterParser(tmp_path).ebs

    assert ebs is not None and ebs.projected is not None
    assert ebs.orbital_names == LOBSTER_ORBITALS
    projected = ebs.projected.to_array()
    assert projected.shape[2:] == (1, 2, 9)
    fe, s = 0, 0
    expected = np.array([[0.1, 0.2, 0.3, 0.4], [0.15, 0.25, 0.35, 0.45]])
    assert projected[:, :, 0, fe, s] == pytest.approx(expected)
    projected[:, :, 0, fe, s] = 0.0
    assert np.all(projected == 0.0)


PDOS_HEADER = "  10.0000 -10.0000   5   0.0000   1.0000"
PDOS_ENERGIES = (-10.0, -5.0, 0.0, 5.0, 10.0)
# s, p_y, p_z and p_x weights of each atom's block, the same on every energy
PDOS_WEIGHTS = {26: (0.1, 0.2, 0.3, 0.4), 8: (0.01, 0.02, 0.03, 0.04)}


def _write_projected_doscar(dirpath) -> None:
    lines = DOSCAR_CONTENT.splitlines()[:6]
    lines += [f" {e:9.4f}   1.0000   0.5000" for e in PDOS_ENERGIES]
    for z, weights in PDOS_WEIGHTS.items():
        lines.append(f"{PDOS_HEADER}; Z= {z}; s p_y p_z p_x")
        lines += [f" {e:9.4f} " + " ".join(f"{w:.4f}" for w in weights) for e in PDOS_ENERGIES]
    (dirpath / "DOSCAR.lobster").write_text("\n".join(lines) + "\n")


def test_dosplot_parametric_colours_a_lobster_dos_by_every_atom(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import pyprocar

    _write_projected_doscar(tmp_path)

    fig, ax = pyprocar.dosplot(
        code="lobster", dirname=str(tmp_path), mode="parametric", orbitals=[1, 2, 3], show=False
    )
    colours = np.asarray(ax.images[0].get_array()).ravel()
    plt.close(fig)

    # p of both atoms over every projection: (0.9 + 0.09) / (1.0 + 0.1)
    np.testing.assert_allclose(colours, [0.9] * 5)


# The lobsterout line and FATBAND header LOBSTER 5.1.1 writes (QE diamond).
# Its labels carry principal numbers.
LOBSTER_5_LOBSTEROUT = """LOBSTER v5.1.1 (g++ 9.2.0)
detecting used PAW program... Quantum Espresso
calculating FatBand for Element: C1 Orbital(s): 2p_x 2s \n"""


def _lobster_5_fatband(orbital: str, weights: tuple[float, float]) -> str:
    return (
        f"# FATBAND for C1 {orbital} NBANDS 2\n"
        "# K-Point 126 : 0.00000000000000 0.00000000000000 0.00000000000000\n"
        f"126          -24.053            {weights[0]:.3f}\n"
        f"126           -2.611            {weights[1]:.3f}\n"
    )


def test_lobster_5_fatband_labels_with_principal_numbers_fill_their_orbital(tmp_path):
    for name, content in {
        "lobsterout": LOBSTER_5_LOBSTEROUT,
        "DOSCAR.lobster": DOSCAR_CONTENT,
        "FATBAND_C1_2s.lobster": _lobster_5_fatband("2s", (0.5, 0.0)),
        "FATBAND_C1_2p_x.lobster": _lobster_5_fatband("2p_x", (0.0, 0.3)),
    }.items():
        (tmp_path / name).write_text(content)

    ebs = LobsterParser(tmp_path).ebs

    assert ebs is not None and ebs.projected is not None and ebs.orbital_names is not None
    projected = ebs.projected.to_array()
    s, p_x = ebs.orbital_names.index("s"), ebs.orbital_names.index("p_x")
    assert projected[0, :, 0, 0, s] == pytest.approx([0.5, 0.0])
    assert projected[0, :, 0, 0, p_x] == pytest.approx([0.0, 0.3])
