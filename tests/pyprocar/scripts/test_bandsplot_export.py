"""bandsplot(export_data_file=...) takes any path with a format suffix (issue #10)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import pyprocar
from tests.utils import DATA_DIR

CALC = DATA_DIR / "examples" / "bands" / "non-spin-polarized"
FERMI = 5.3017


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")  # a figure an earlier test on this worker left open is not ours
    yield
    plt.close("all")


@pytest.mark.data
@pytest.mark.parametrize(
    ("export_data_file", "append_mode", "written"),
    [
        # The issue's traceback: "./bands.dat".split(".") has three parts, not two.
        ("./bands.dat", True, "bands_plain.dat"),
        ("./out.d/bands.dat", True, "out.d/bands_plain.dat"),
        (Path("out.d/bands.csv"), False, "out.d/bands.csv"),
    ],
)
def test_export_writes_next_to_the_given_path(
    tmp_path: Path, monkeypatch, export_data_file, append_mode, written
) -> None:
    if not CALC.exists():
        pytest.skip("fixture bands/non-spin-polarized not downloaded")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out.d").mkdir()

    pyprocar.bandsplot(
        code="vasp",
        dirname=str(CALC),
        mode="plain",
        fermi=FERMI,
        export_data_file=export_data_file,
        export_append_mode=append_mode,
        show=False,
    )

    header = (tmp_path / written).read_text().splitlines()[0]
    assert header.split("," if written.endswith(".csv") else " ")[0] == "kpath_values"


def test_unknown_suffix_raises_before_reading_or_plotting(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.csv, \.txt, \.dat or \.json"):
        pyprocar.bandsplot(
            code="vasp",
            dirname=str(tmp_path / "no-calculation-here"),
            export_data_file=str(tmp_path / "bands.xlsx"),
            show=False,
        )

    assert plt.get_fignums() == []
    assert list(tmp_path.iterdir()) == []
