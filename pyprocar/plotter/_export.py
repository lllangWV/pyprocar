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


def write_table(path: str | Path, columns: Mapping[str, npt.ArrayLike]) -> None:
    """Write equal-length ``columns`` in order: a JSON object of lists, or a delimited table."""
    path = export_path(path)
    if path.suffix == ".json":
        lists = {name: np.asarray(values).tolist() for name, values in columns.items()}
        path.write_text(json.dumps(lists), encoding="utf-8")
        return
    table = pd.DataFrame({name: np.asarray(values) for name, values in columns.items()})
    table.to_csv(path, sep=DELIMITERS[path.suffix], index=False)
