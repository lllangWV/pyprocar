"""Write plotted curves as one table, in the format the file's suffix names."""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

DELIMITERS = {".csv": ",", ".txt": "\t", ".dat": " "}


def export_path(path: str | Path) -> Path:
    """``path`` as a Path, or a ValueError when its suffix names no export format."""
    path = Path(path)
    if path.suffix not in (*DELIMITERS, ".json"):
        raise ValueError(
            "An export file must end in .csv, .txt, .dat or .json, which sets its format;"
            + f" got {str(path)!r}."
        )
    return path


def recorded_columns(values: Mapping[str, npt.ArrayLike | None]) -> dict[str, np.ndarray]:
    """The non-empty arrays a plotter recorded, or a ValueError when it recorded none."""
    if not values:
        raise ValueError("No values recorded. Plot first before exporting.")
    arrays = {key: np.atleast_1d(value) for key, value in values.items() if value is not None}
    return {key: array for key, array in arrays.items() if array.size > 0}


def write_table(path: str | Path, columns: Mapping[str, npt.ArrayLike]) -> None:
    """Write equal-length ``columns`` in order: a JSON object of lists, or a delimited table."""
    path = export_path(path)
    if path.suffix == ".json":
        lists = {name: np.asarray(values).tolist() for name, values in columns.items()}
        path.write_text(json.dumps(lists), encoding="utf-8")
        return
    table = pd.DataFrame({name: np.asarray(values) for name, values in columns.items()})
    table.to_csv(path, sep=DELIMITERS[path.suffix], index=False)
