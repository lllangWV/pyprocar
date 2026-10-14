import logging
from functools import cached_property
from pathlib import Path
from typing import TypedDict

import numpy as np

from pyprocar.core.atomic_orbital_index import SHELL_LETTERS, OrbitalIndexer

logger = logging.getLogger(__name__)

_LM_ORBITALS = OrbitalIndexer().flat_conventional
# The orbitals a VASP DOS projects onto, by their count per atom: s, p, d (and f)
# with LORBIT=10, or the 9 (16) lm orbitals with LORBIT=11.
DOS_ORBITAL_NAMES: dict[int, tuple[str, ...]] = {
    3: SHELL_LETTERS[:3],
    4: SHELL_LETTERS,
    9: tuple(_LM_ORBITALS[:9]),
    16: tuple(_LM_ORBITALS[:16]),
}
ORBITAL_COUNTS = tuple(DOS_ORBITAL_NAMES)
HEADER_LINES = 6
_VASP_ORBITALS = (
    "VASP projects onto s, p, d (and f) with LORBIT=10, or onto the 9 (16) lm orbitals"
    + " with LORBIT=11"
)


def _fits_collinear(n_columns: int) -> bool:
    return n_columns in ORBITAL_COUNTS


def _fits_noncollinear(n_columns: int) -> bool:
    return n_columns % 4 == 0 and n_columns // 4 in ORBITAL_COUNTS


def dos_orbital_names(n_orbitals: int) -> tuple[str, ...]:
    """The names of the ``n_orbitals`` orbitals a VASP DOS projects onto per atom."""
    if n_orbitals not in DOS_ORBITAL_NAMES:
        raise ValueError(f"The DOS projects onto {n_orbitals} orbitals per atom; {_VASP_ORBITALS}.")
    return DOS_ORBITAL_NAMES[n_orbitals]


class HeaderLine1(TypedDict):
    natoms: int
    natoms_check: int
    lorbit_flag: int
    ncdij: int


class HeaderLine6(TypedDict):
    emax: float
    emin: float
    nedos: int
    efermi: float
    weight: float


class Doscar:
    """
    A class to parse the DOSCAR file

    Parameters
    ----------
    filepath : str | Path, optional
        The DOSCAR filepath, by default "DOSCAR"
    file_str : str, optional
        The DOSCAR file content as a string, by default None
    """

    def __init__(
        self,
        filepath: str | Path | None = None,
        file_str: str = "",
    ) -> None:
        logger.info("Initializing DOSCAR parser for %s", filepath)
        self._filepath: str | Path | None = filepath
        self._file_str: str = file_str

    @classmethod
    def from_str(cls, input_str: str) -> "Doscar":
        return cls(file_str=input_str)

    @property
    def filepath(self) -> Path | None:
        if self._filepath is None:
            return None
        return Path(self._filepath)

    @cached_property
    def file_str(self) -> str:
        if self._file_str == "" and self.filepath is not None:
            with open(file=self.filepath) as rf:
                file_str = rf.read()
        elif self._file_str != "":
            file_str = self._file_str
        else:
            raise ValueError("No file path or file string provided")
        return file_str

    @cached_property
    def _source(self) -> str:
        return str(self.filepath) if self.filepath is not None else "text"

    @cached_property
    def _all_lines(self) -> list[str]:
        return self.file_str.splitlines()

    @cached_property
    def has_dos(self) -> bool:
        """Whether a DOS follows the header; a run killed early leaves none."""
        return any(line.strip() for line in self._all_lines[HEADER_LINES:])

    @cached_property
    def lines(self) -> list[str]:
        """The file's lines, or a ValueError naming it when no DOS follows its header."""
        if not self.has_dos:
            raise ValueError(
                f"The DOSCAR {self._source} holds no DOS: it has {len(self._all_lines)} lines,"
                + f" and VASP writes the DOS after a {HEADER_LINES}-line header at the end of"
                + " a run. Rerun the calculation, or keep its vasprun.xml next to the DOSCAR."
            )
        return self._all_lines

    # ---------------------------
    # === HEADER SECTION ===
    # ---------------------------

    @cached_property
    def header_line1(self) -> HeaderLine1:
        nions, nions2, lorbit_flag, ncdij = map(int, self.lines[0].split())
        return {
            "natoms": nions,
            "natoms_check": nions2,
            "lorbit_flag": lorbit_flag,
            "ncdij": ncdij,
        }

    @cached_property
    def header_line2(self):
        vals = list(map(float, self.lines[1].split()))
        return {
            "volume": vals[0],
            "a": vals[1],
            "b": vals[2],
            "c": vals[3],
            "potim": vals[4],
        }

    @cached_property
    def tebeg(self):
        return float(self.lines[2].split()[0])

    @cached_property
    def car_label(self):
        return self.lines[3].strip()

    @cached_property
    def system_name(self):
        return self.lines[4].strip()

    @cached_property
    def header_line6(self) -> HeaderLine6:
        emax, emin, nedos, efermi, weight = self.lines[5].split()
        return {
            "emax": float(emax),
            "emin": float(emin),
            "nedos": int(nedos),
            "efermi": float(efermi),
            "weight": float(weight),
        }

    # convenience properties
    @cached_property
    def natoms(self) -> int:
        return self.header_line1["natoms"]

    @cached_property
    def nedos(self) -> int:
        return self.header_line6["nedos"]

    @cached_property
    def efermi(self) -> float:
        return self.header_line6["efermi"]

    # ---------------------------
    # === TOTAL DOS ===
    # ---------------------------

    @cached_property
    def is_spin_pol(self) -> bool:
        return self._raw_total_dos.shape[1] == 5

    @cached_property
    def _raw_total_dos(self) -> np.ndarray:
        """
        For ISPIN=1 → shape (NEDOS, 3) [E, DOS, IntDOS]
        For ISPIN=2 → shape (NEDOS, 5) [E, DOSup, DOSdown, IntDOSup, IntDOSdown]
        """
        block = self._read_block(HEADER_LINES, "total DOS")
        if block.shape[1] not in (3, 5):
            raise ValueError(
                f"The DOSCAR {self._source} total DOS has {block.shape[1]} columns; VASP writes 3"
                + " (energy, DOS, integrated DOS) or 5 for a spin-polarized run."
            )
        return block

    def _read_block(self, start: int, what: str) -> np.ndarray:
        """The ``nedos`` rows from line ``start`` as one array, or a ValueError saying why not."""
        block = self.lines[start : start + self.nedos]
        rows = [line.split() for line in block]
        if len(rows) < self.nedos or len({len(row) for row in rows}) != 1:
            raise ValueError(
                f"The DOSCAR {self._source} {what} block from line {start + 1} is not"
                + f" {self.nedos} rows of equal length; the file is truncated or not a VASP DOSCAR."
            )
        return np.array(rows, dtype=float)

    @cached_property
    def total_dos(self) -> np.ndarray:
        return self._raw_total_dos[:, 1:3] if self.is_spin_pol else self._raw_total_dos[:, 1:2]

    @cached_property
    def integrated_dos(self) -> np.ndarray:
        return self._raw_total_dos[:, 3:] if self.is_spin_pol else self._raw_total_dos[:, 2:]

    @cached_property
    def energies(self) -> np.ndarray:
        return self._raw_total_dos[:, 0]

    # ---------------------------
    # === PROJECTED DOS (PDOS) ===
    # ---------------------------

    def projected(self, non_collinear: bool | None = None) -> np.ndarray | None:
        """
        Return PDOS in shape (n_energies, n_spins, n_atoms, n_orbitals), or None
        when the DOSCAR has no projected blocks (LORBIT unset).

        ``non_collinear`` is LNONCOLLINEAR of the run (from the OUTCAR or PROCAR). It is
        needed only to read 16 columns per atom, which are either the 16 lm orbitals of a
        collinear run or s, p, d, f times (total, mx, my, mz) of a non-collinear one.
        """
        nedos, natoms = self.nedos, self.natoms
        start = HEADER_LINES + nedos
        if not any(line.strip() for line in self.lines[start:]):
            return None

        def columns(atom: int) -> np.ndarray:
            first_row = start + 1 + atom * (nedos + 1)
            return self._read_block(first_row, f"atom {atom + 1} projected DOS")[:, 1:]

        first = columns(0)
        n_columns = first.shape[1]
        n_spins = self._projected_spins(n_columns, non_collinear)
        projected = np.empty((natoms, nedos, n_columns))
        projected[0] = first
        for atom in range(1, natoms):
            block = columns(atom)
            if block.shape != first.shape:
                raise ValueError(
                    f"The DOSCAR {self._source} has {block.shape[1]} projected columns for atom"
                    + f" {atom + 1} and {first.shape[1]} for atom 1; VASP writes the same for every"
                    + " atom."
                )
            projected[atom] = block
        # VASP writes each orbital's spin columns together: s_up s_down py_up ... for
        # ISPIN=2 and s_tot s_mx s_my s_mz py_tot ... for non-collinear runs.
        return projected.reshape(natoms, nedos, n_columns // n_spins, n_spins).transpose(1, 3, 0, 2)

    @cached_property
    def needs_lnoncollinear(self) -> bool:
        """Whether the projected columns read as both a collinear and a non-collinear
        layout (16 per atom with one total DOS channel), so that ``projected`` needs
        LNONCOLLINEAR to tell them apart."""
        first_row = self.lines[HEADER_LINES + self.nedos + 1 : HEADER_LINES + self.nedos + 2]
        if self.is_spin_pol or not first_row:
            return False
        n_columns = len(first_row[0].split()) - 1
        return _fits_collinear(n_columns) and _fits_noncollinear(n_columns)

    def _projected_spins(self, n_columns: int, non_collinear: bool | None) -> int:
        if self.is_spin_pol:
            if n_columns % 2:
                raise ValueError(
                    f"The DOSCAR {self._source} has {n_columns} projected columns per atom; a"
                    + " spin-polarized run writes an up and a down column per orbital."
                )
            return 2
        fits_collinear = _fits_collinear(n_columns)
        fits_noncollinear = _fits_noncollinear(n_columns)
        if non_collinear is None:
            if fits_collinear and fits_noncollinear:
                raise ValueError(
                    f"The DOSCAR {self._source} has {n_columns} projected columns per atom:"
                    + f" {n_columns} orbitals of a collinear run or {n_columns // 4} orbitals"
                    + " times (total, mx, my, mz) of a non-collinear one. Keep the run's OUTCAR"
                    + " (LNONCOLLINEAR) or PROCAR, which tell them apart, next to the DOSCAR."
                )
            non_collinear = fits_noncollinear
        if non_collinear and fits_noncollinear:
            return 4
        if not non_collinear and fits_collinear:
            return 1
        raise ValueError(
            f"The DOSCAR {self._source} has {n_columns} projected columns per atom, which no"
            + f" {'non-collinear' if non_collinear else 'non-spin-polarized'} VASP run writes:"
            + f" {_VASP_ORBITALS}, times 4 (total, mx, my, mz) when non-collinear."
        )
