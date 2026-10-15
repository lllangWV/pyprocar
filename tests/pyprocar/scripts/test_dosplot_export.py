"""dosplot(export_data_file=...) writes the curves it draws (issue #10)."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection

import pyprocar
from tests.utils import DATA_DIR

DOS_DIR = DATA_DIR / "examples" / "dos"
FERMI = 5.3017


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")  # a figure an earlier test on this worker left open is not ours
    yield
    plt.close("all")


def _calc(name: str) -> str:
    calc = DOS_DIR / name
    if not calc.exists():
        pytest.skip(f"fixture dos/{name} not downloaded")
    return str(calc)


def read_table(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".json":
        return {k: np.asarray(v, dtype=float) for k, v in json.loads(path.read_text()).items()}
    delimiter = {".csv": ",", ".txt": "\t", ".dat": " "}[path.suffix]
    with path.open(newline="") as handle:
        header, *rows = list(csv.reader(handle, delimiter=delimiter))
    values = np.array(rows, dtype=float)
    return {name: values[:, i] for i, name in enumerate(header)}


def drawn_lines(ax) -> list[np.ndarray]:
    return [
        np.asarray(line.get_ydata(), dtype=float)
        for line in ax.lines
        if not str(line.get_label()).startswith("_")
    ]


@pytest.mark.data
def test_reporters_call_writes_the_drawn_curves(tmp_path: Path, monkeypatch) -> None:
    # The issue's call: export_data_file="./dos_data.dat", which split(".") could not unpack.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out.d").mkdir()

    _, ax = pyprocar.dosplot(
        code="vasp",
        dirname=_calc("non-spin-polarized"),
        mode="overlay_orbitals",
        fermi=FERMI,
        elimit=[-2, 2],
        atoms=[1],
        export_data_file="./out.d/dos_data.dat",
        show=False,
    )

    table = read_table(tmp_path / "out.d" / "dos_data.dat")
    assert list(table) == ["energy", "total", "s", "p", "d"]
    np.testing.assert_array_equal(table["energy"], ax.lines[0].get_xdata())
    for column, y in zip(list(table)[1:], drawn_lines(ax), strict=True):
        np.testing.assert_array_equal(table[column], y)
    # DOSCAR row "5.305  0.1237E+01 ..." and V's d columns 3 x 0.3387 + 2 x 0.4740E-03.
    row = int(np.argmin(np.abs(table["energy"] - (5.305 - FERMI))))
    assert table["total"][row] == pytest.approx(1.237, rel=1e-3)
    assert table["d"][row] == pytest.approx(3 * 0.3387 + 2 * 0.4740e-3, rel=1e-3)


@pytest.mark.data
@pytest.mark.parametrize("suffix", [".csv", ".txt", ".json"])
def test_spin_channels_get_a_column_each(tmp_path: Path, suffix: str) -> None:
    out = tmp_path / f"dos{suffix}"

    _, ax = pyprocar.dosplot(
        code="vasp",
        dirname=_calc("spin-polarized"),
        mode="overlay_species",
        fermi=FERMI,
        export_data_file=str(out),
        show=False,
    )

    table = read_table(out)
    names = ["total", "O", "Sr", "V"]
    assert list(table) == ["energy", *(f"{n}_spin-{c}" for n in names for c in (0, 1))]
    # The plot mirrors spin-down below zero; the file keeps the DOS's own sign.
    for column, y in zip(list(table)[1:], drawn_lines(ax), strict=True):
        np.testing.assert_array_equal(table[column], -y if column.endswith("spin-1") else y)
    assert table["total_spin-1"].min() >= 0


@pytest.mark.data
def test_stack_exports_each_component_unstacked(tmp_path: Path) -> None:
    out = tmp_path / "dos.csv"

    _, ax = pyprocar.dosplot(
        code="vasp",
        dirname=_calc("non-spin-polarized"),
        mode="stack_orbitals",
        fermi=FERMI,
        atoms=[1],
        export_data_file=str(out),
        show=False,
    )

    table = read_table(out)
    assert list(table) == ["energy", "total", "s", "p", "d"]
    n = len(table["energy"])
    tops = [c.get_paths()[0].vertices[1 : n + 1, 1] for c in ax.collections]
    assert all(isinstance(c, PolyCollection) for c in ax.collections)
    np.testing.assert_allclose(table["s"], tops[0])
    np.testing.assert_allclose(table["s"] + table["p"], tops[1])
    np.testing.assert_allclose(table["s"] + table["p"] + table["d"], tops[2])


@pytest.mark.data
def test_parametric_exports_the_projection_it_colours_by(tmp_path: Path) -> None:
    out = tmp_path / "dos.json"

    _, ax = pyprocar.dosplot(
        code="vasp",
        dirname=_calc("non-spin-polarized"),
        mode="parametric",
        fermi=FERMI,
        atoms=[1],
        orbitals=[4, 5, 6, 7, 8],
        export_data_file=str(out),
        show=False,
    )

    table = read_table(out)
    assert list(table) == ["energy", "total", "projection"]
    np.testing.assert_array_equal(table["projection"], np.ravel(ax.images[0].get_array()))


def test_unknown_suffix_raises_before_reading_or_plotting(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.csv, \.txt, \.dat or \.json"):
        pyprocar.dosplot(
            code="vasp",
            dirname=str(tmp_path / "no-calculation-here"),
            export_data_file=str(tmp_path / "dos.xlsx"),
            show=False,
        )

    assert plt.get_fignums() == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.data
def test_items_entries_for_one_species_keep_a_column_each(tmp_path: Path) -> None:
    out = tmp_path / "dos.dat"

    _, ax = pyprocar.dosplot(
        code="vasp",
        dirname=_calc("non-spin-polarized"),
        mode="overlay",
        fermi=FERMI,
        items=[{"V": ["d"]}, {"V": ["p"]}],
        export_data_file=str(out),
        show=False,
    )

    table = read_table(out)
    assert list(table) == ["energy", "total", "V", "V-2"]
    for column, y in zip(list(table)[1:], drawn_lines(ax), strict=True):
        np.testing.assert_array_equal(table[column], y)
