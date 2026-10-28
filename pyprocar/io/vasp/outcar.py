import logging
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, override

import numpy as np
import spglib
from numpy.typing import NDArray

from pyprocar.core.kpoints import KGRID_MODE, get_kpoints_from_kgrid

logger = logging.getLogger(__name__)

_PARAMETER_LISTING = "Startparameter for this run:"
_LISTED_FLAG = re.compile(
    r"^[ \t]*(LNONCOLLINEAR|LSORBIT)[ \t]*=[ \t]*(T|F|\.TRUE\.|\.FALSE\.)(?![\w.])",
    re.MULTILINE | re.IGNORECASE,
)

# VASP's Schoenflies point-group names and the international symbols spglib gives them.
_POINT_GROUP_SYMBOLS = {
    "C_1": "1", "S_2": "-1", "C_2": "2", "C_1h": "m", "C_2h": "2/m", "D_2": "222",
    "C_2v": "mm2", "D_2h": "mmm", "C_4": "4", "S_4": "-4", "C_4h": "4/m", "D_4": "422",
    "C_4v": "4mm", "D_2d": "-42m", "D_4h": "4/mmm", "C_3": "3", "S_6": "-3", "D_3": "32",
    "C_3v": "3m", "D_3d": "-3m", "C_6": "6", "C_3h": "-6", "C_6h": "6/m", "D_6": "622",
    "C_6v": "6mm", "D_3h": "-6m2", "D_6h": "6/mmm", "T": "23", "T_h": "m-3", "O": "432",
    "T_d": "-43m", "O_h": "m-3m",
}  # fmt: skip

_FLOAT = r"[-+]?\d+\.\d+"
_POINT_GROUP = re.compile(
    r"point symmetry\s+(?P<site>\S+?)\s*\."
    + r"(?:\s*\n\s*The point group associated with its full space group is\s+(?P<full>\S+?)\s*\.)?"
    + r"|NOSYMM:.*point group\s+(?P<none>\S+?)\s*\."
)
_IRREDUCIBLE = re.compile(
    r"Found\s+(\d+)\s+irreducible k-points:\s*\n\s*\n\s*Following reciprocal coordinates:\s*\n.*\n"
)
_SHIFT = re.compile(
    r"Shift w\.r\.t\. Gamma in fractional coordinates \(k-lattice\)\s*\n"
    + rf"\s*({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})"
)
# VASP 6 echoes each INCAR tag on its own line under " INCAR:"; VASP 5 lists only POTCARs there.
_INCAR_ECHO = re.compile(r"^ INCAR:\n((?:[ \t]+\w+[ \t]*=.*\n)+)", re.MULTILINE)
_DEFAULT_SYMPREC = 1e-5


@dataclass(frozen=True)
class _MeshReduction:
    """What an OUTCAR states about how VASP reduced its automatic k-mesh.

    Invariants: kpoints and multiplicities have one row per irreducible k-point, and the
    multiplicities are positive integers, not all 1. They sum to prod(mesh), or to more when
    the operations carry the mesh off itself and VASP adds the images. moments, when known,
    has one row per atom of the cell: a Cartesian vector for a non-collinear run, one value
    for a collinear spin-polarized run.
    """

    cell: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]]  # spglib's cell
    point_group: str | None
    n_operations: int | None
    time_reversal: bool
    symprec: float
    moments: NDArray[np.float64] | None
    mesh: NDArray[np.int_]
    shift: NDArray[np.float64]
    kpoints: NDArray[np.float64]
    multiplicities: NDArray[np.int_]

    @property
    def grain(self) -> int:
        """Mesh points and their images under integer rotations are multiples of 1/grain."""
        return 2 * int(np.lcm.reduce(self.mesh))


def _last(pattern: str | re.Pattern[str], text: str) -> re.Match[str] | None:
    matches = list(re.finditer(pattern, text))
    return matches[-1] if matches else None


def _incar_tags(text: str) -> dict[str, str] | None:
    """The INCAR tags VASP echoes at the top of the OUTCAR, or None when it echoes none."""
    echo = _INCAR_ECHO.search(text)
    if echo is None:
        return None
    tags: dict[str, str] = {}
    for line in echo.group(1).splitlines():
        key, _, value = line.partition("=")
        tags[key.strip().upper()] = re.split(r"[!#]", value)[0].strip()
    return tags


def _incar_numbers(value: str) -> list[float] | None:
    """INCAR numbers with VASP's ``n*x`` repeats spelled out and Fortran ``d`` exponents read,
    or None when one is not a number."""
    numbers: list[float] = []
    try:
        for token in value.lower().replace("d", "e").split():
            count, star, number = token.partition("*")
            numbers.extend([float(number)] * int(count) if star else [float(token)])
    except ValueError:
        return None
    return numbers


def _initial_moments(
    incar: Mapping[str, str] | None,
    n_atoms: int,
    non_collinear: bool,
    spin_polarized: bool,
) -> NDArray[np.float64] | None:
    """The initial magnetic moments VASP analysed the symmetry with, one row per atom.

    Non-collinear: Cartesian vectors, turned from the SAXIS frame MAGMOM is written in.
    Collinear spin-polarized: one value per atom. None for a non-magnetic run, when the
    OUTCAR echoes no INCAR (VASP 5), or when MAGMOM or SAXIS does not parse.
    """
    if incar is None or not (non_collinear or spin_polarized):
        return None
    width = 3 if non_collinear else 1
    if "MAGMOM" in incar:
        magmom = _incar_numbers(incar["MAGMOM"])
        if magmom is None or len(magmom) != width * n_atoms:
            return None
        moments = np.array(magmom).reshape(n_atoms, width)
    else:
        # VASP's default MAGMOM: 1 per atom, (1, 1, 1) per atom when non-collinear.
        moments = np.ones((n_atoms, width))
    if not non_collinear:
        return moments[:, 0]
    saxis = _incar_numbers(incar.get("SAXIS", "0 0 1"))
    if saxis is None or len(saxis) != 3:
        return None
    return moments @ _saxis_frame(np.array(saxis)).T


def _saxis_frame(saxis: NDArray[np.float64]) -> NDArray[np.float64]:
    """The rotation that takes a moment written along SAXIS's axes to Cartesian axes, from
    the formula on the VASP wiki's SAXIS page."""
    alpha = np.arctan2(saxis[1], saxis[0])
    beta = np.arctan2(np.hypot(saxis[0], saxis[1]), saxis[2])
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    return np.array(
        [
            [cb * ca, -sa, sb * ca],
            [cb * sa, ca, sb * sa],
            [-sb, 0.0, cb],
        ]
    )


def _mesh_reduction(
    text: str, kgrid: tuple[int | None, ...], non_collinear: bool | None
) -> _MeshReduction | None:
    """The record, or None when the OUTCAR shows no automatic mesh reduced by symmetry (band
    paths, explicit lists, full meshes) or leaves out a fact the fit needs."""
    irreducible = _last(_IRREDUCIBLE, text)
    shift = _last(_SHIFT, text)
    per_type = re.search(r"ions per type\s*=\s*([\d\s]+?)\n", text)
    if None in kgrid or non_collinear is None or not (irreducible and shift and per_type):
        return None
    mesh = np.array(kgrid, dtype=int)
    rows = text[irreducible.end() :].splitlines()[: int(irreducible.group(1))]
    listed = np.array([row.split()[:4] for row in rows], dtype=float)
    multiplicities = np.rint(listed[:, 3]).astype(int)
    if (
        np.any(np.abs(listed[:, 3] - multiplicities) > 1e-3)
        or np.any(multiplicities < 1)
        or np.all(multiplicities == 1)
        or multiplicities.sum() < np.prod(mesh)
    ):
        return None

    # The cell VASP analysed is printed after "ions per type"; earlier blocks can be the
    # primitive cell PRICEL found.
    counts = [int(n) for n in per_type.group(1).split()]
    lattice_block = re.compile(r"direct lattice vectors.*\n((?:.*\n){3})").search(
        text, per_type.end()
    )
    position_block = re.compile(
        r"position of ions in fractional coordinates \(direct lattice\)\s*\n"
        + rf"((?:.*\n){{{sum(counts)}}})"
    ).search(text, per_type.end())
    if lattice_block is None or position_block is None:
        return None
    lattice, positions = (
        np.array(
            [re.findall(_FLOAT, line)[:3] for line in block.group(1).splitlines()],
            dtype=float,
        )
        for block in (lattice_block, position_block)
    )

    point_group = None
    if named := _last(_POINT_GROUP, text):
        name = named.group("full") or named.group("site") or named.group("none")
        if name not in _POINT_GROUP_SYMBOLS:
            logger.info("Unknown point group %s in the OUTCAR", name)
            return None
        point_group = _POINT_GROUP_SYMBOLS[name]
    count = _last(r"INISYM returns: Found\s+(\d+)\s+space group operations", text)

    incar = _incar_tags(text)
    symprec = _incar_numbers((incar or {}).get("SYMPREC", str(_DEFAULT_SYMPREC)))
    if not symprec:
        return None
    listing = text[text.find(_PARAMETER_LISTING) :]
    spin_polarized = re.search(r"^\s*ISPIN\s*=\s*2\b", listing, re.MULTILINE) is not None

    return _MeshReduction(
        cell=(lattice, positions, np.repeat(np.arange(len(counts)), counts)),
        point_group=point_group,
        n_operations=int(count.group(1)) if count else None,
        time_reversal=not non_collinear,
        symprec=symprec[0],
        moments=_initial_moments(incar, sum(counts), non_collinear, spin_polarized),
        mesh=mesh,
        shift=np.array(shift.groups(), dtype=float),
        kpoints=listed[:, :3],
        multiplicities=multiplicities,
    )


def _subgroups(rotations: NDArray[np.int_], order: int | None) -> list[NDArray[np.int_]]:
    """The subgroups of a finite group of integer matrices with ``order`` elements (all of
    them for None), each with the identity first."""
    keys = [rotation.tobytes() for rotation in rotations]
    index = {key: i for i, key in enumerate(keys)}
    product = [[index[(a @ b).tobytes()] for b in rotations] for a in rotations]
    identity = index[np.eye(3, dtype=rotations.dtype).tobytes()]
    first = sorted(range(len(rotations)), key=lambda i: (i != identity, i))
    if order == len(rotations):
        return [rotations[first]]

    def closure(members: frozenset[int]) -> frozenset[int]:
        group, frontier = set(members), list(members)
        while frontier:
            a = frontier.pop()
            for b in list(group):
                for c in (product[a][b], product[b][a]):
                    if c not in group:
                        group.add(c)
                        frontier.append(c)
        return frozenset(group)

    found = {frozenset({identity})}
    frontier = list(found)
    while frontier:
        group = frontier.pop()
        for g in range(len(rotations)):
            if g in group:
                continue
            bigger = closure(group | {g})
            # A subgroup's subgroups have orders dividing its own (Lagrange).
            if (order is None or order % len(bigger) == 0) and bigger not in found:
                found.add(bigger)
                frontier.append(bigger)
    return [
        rotations[[i for i in first if i in group]]
        for group in found
        if order is None or len(group) == order
    ]


def _keeps_k_lattice(k_rotations: NDArray[np.int_], mesh: NDArray[np.int_]) -> NDArray[np.bool_]:
    """Which rotations map the lattice of mesh steps onto itself."""
    scaled = mesh[:, np.newaxis] * k_rotations / mesh[np.newaxis, :]
    return np.all(np.abs(scaled - np.rint(scaled)) < 1e-9, axis=(1, 2))


def _test_points(reduction: _MeshReduction) -> set[int] | None:
    """The keys of the k-points VASP's IBZKPT reduces: the mesh, and, when every operation of
    the reciprocal lattice keeps the mesh's lattice, their images under those operations
    (VASP's "blow-up", which adds the points a shifted mesh is carried to; an unshifted mesh
    goes onto itself)."""
    mesh = get_kpoints_from_kgrid(
        kgrid=tuple(reduction.mesh),
        kshift=tuple(reduction.shift),
        mode=KGRID_MODE.GAMMA,
    )
    if np.any(reduction.shift):
        point = (reduction.cell[0], np.zeros((1, 3)), np.zeros(1, dtype=int))
        lattice = spglib.get_symmetry_dataset(point, symprec=reduction.symprec)  # ty: ignore[invalid-argument-type]
        if lattice is None:
            return None
        rotations = _k_rotations(lattice.rotations)
        if np.all(_keeps_k_lattice(rotations, reduction.mesh)):
            mesh = np.einsum("oij,kj->koi", rotations.astype(float), mesh).reshape(-1, 3)
    keys = _snapped_keys(mesh, reduction.grain)
    return None if keys is None else set(keys.tolist())


def _reproduces_irreducible_kpoints(
    k_rotations: NDArray[np.int_],
    reduction: _MeshReduction,
    test_points: set[int],
) -> bool:
    """Whether the orbits of the listed k-points, cut to VASP's test points, have exactly the
    listed multiplicities, are disjoint, and hold every test point.

    VASP's IBZKPT (mkpoints.F) adds time reversal (k -> -k) for a collinear run only. It
    merges a test point into an earlier irreducible k-point when one operation maps it there,
    so an orbit counts the distinct images that are test points, whether or not the operation
    keeps the mesh's lattice.
    """
    operations = k_rotations.astype(float)
    if reduction.time_reversal:
        operations = np.concatenate([operations, -operations])
    images = np.einsum("oij,kj->koi", operations, reduction.kpoints)
    image_keys = _snapped_keys(images, reduction.grain)
    if image_keys is None:
        return False
    covered: set[int] = set()
    for orbit_keys, multiplicity in zip(image_keys, reduction.multiplicities, strict=True):
        orbit = test_points.intersection(orbit_keys.tolist())
        if len(orbit) != multiplicity or not covered.isdisjoint(orbit):
            return False
        covered |= orbit
    return covered == test_points


def _snapped_keys(kpoints: NDArray[np.floating], grain: int) -> NDArray[np.int64] | None:
    """One integer per fractional k modulo reciprocal lattice vectors, or None when a k is not
    a multiple of 1/grain (OUTCAR coordinates carry 6 decimals)."""
    steps = kpoints * grain
    if np.any(np.abs(steps - np.rint(steps)) > 0.05):
        return None
    cells = np.mod(np.rint(steps).astype(np.int64), grain)
    return cells @ np.array([grain * grain, grain, 1], dtype=np.int64)


def _fitting_operations(reduction: _MeshReduction) -> list[NDArray[np.int_]]:
    """Every subgroup of the crystal's point operations that fits all the OUTCAR states, as
    k-space rotations (k' = R k with R = inv(W).T for the real-space W), identity first.

    A subgroup fits when spglib names its point group as VASP did, it has VASP's operation
    count, and its orbits reproduce the irreducible k-points and their multiplicities. The
    crystal's operations come from spglib on the cell VASP analysed, coloured by POTCAR type,
    at the run's SYMPREC. When several subgroups fit (conjugate ones, such as the four S_6 of
    O_h), only those whose every operation keeps VASP's initial magnetic moments remain:
    VASP's own group is the crystal operations that keep them.
    """
    with warnings.catch_warnings():
        # spglib warns on every call while its process-wide OLD_ERROR_HANDLING is on; the
        # None returns below are the error check it asks for.
        warnings.filterwarnings("ignore", "Set OLD_ERROR_HANDLING", DeprecationWarning)
        return _fitting_subgroups(reduction)


def _fitting_subgroups(reduction: _MeshReduction) -> list[NDArray[np.int_]]:
    dataset = spglib.get_symmetry_dataset(reduction.cell, symprec=reduction.symprec)  # ty: ignore[invalid-argument-type]
    test_points = _test_points(reduction)
    if dataset is None or test_points is None:
        return []
    crystal = np.unique(np.asarray(dataset.rotations, dtype=int), axis=0)
    fitting: list[NDArray[np.int_]] = []
    for group in _subgroups(crystal, reduction.n_operations):
        if reduction.point_group is not None:
            named = spglib.get_pointgroup(group)
            if named is None or named[0].strip() != reduction.point_group:
                continue
        if _reproduces_irreducible_kpoints(_k_rotations(group), reduction, test_points):
            fitting.append(group)
    if len(fitting) > 1 and reduction.moments is not None:
        keeping = _moment_keeping_rotations(reduction.cell, reduction.moments, reduction.symprec)
        fitting = [group for group in fitting if all(w.tobytes() in keeping for w in group)]
    return [_k_rotations(group) for group in fitting]


def _k_rotations(rotations: NDArray[np.integer]) -> NDArray[np.int_]:
    return np.rint(np.linalg.inv(rotations).transpose(0, 2, 1)).astype(int)


def _moment_keeping_rotations(
    cell: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]],
    moments: NDArray[np.float64],
    symprec: float,
) -> set[bytes]:
    """The real-space rotations, as ``tobytes`` keys, of the crystal operations that keep the
    initial moments without time reversal. A non-collinear moment is an axial vector."""
    found = spglib.get_magnetic_symmetry(
        (*cell, moments),  # ty: ignore[invalid-argument-type]
        symprec=symprec,
        is_axial=moments.ndim == 2,
    )
    if found is None:
        return set()
    return {
        np.asarray(rotation, dtype=int).tobytes()
        for rotation, reversed_ in zip(found["rotations"], found["time_reversals"], strict=True)
        if not reversed_
    }


class Outcar(Mapping[str, Any]):
    """
    A class to parse the OUTCAR file from a VASP run and extract electronic structure data.

    The OUTCAR file provides detailed output of a VASP run,
    including a summary of used input parameters,
    information about electronic steps and KS-eigenvalues,
    stress tensors, forces on atoms, local charges
    and magnetic moments, and dielectric properties. The amount of output written onto the
    OUTCAR file can
    be chosen by modifying the NWRITE tag in the INCAR file.

    The Outcar class acts as a Mapping, providing key-value access to the variables parsed from
    the OUTCAR file.
    """

    def __init__(self, filepath: str | Path | None = None, file_str: str = ""):
        """
        Constructor method to initialize an Outcar object. Reads the file specified
        by filename and stores its content.

        Parameters
        ----------
        filename : Union[str, Path], optional
            The OUTCAR filename. If not provided, defaults to "OUTCAR".
        """
        self._file_str: str = file_str
        self._filepath: str | Path | None = filepath

    @classmethod
    def from_str(cls, input: str) -> "Outcar":
        return cls(file_str=input)

    @property
    def filepath(self) -> Path | None:
        if self._filepath is None:
            return None
        return Path(self._filepath)

    @cached_property
    def file_str(self) -> str:
        if self._file_str == "" and self.filepath is not None:
            with open(self.filepath) as file_stream:
                self._file_str = file_stream.read()
        elif self._file_str == "" and self.filepath is None:
            raise ValueError("No file path or file string provided")
        return self._file_str

    def _get_axes_nk(self):
        """
        n_kx

        Returns
        -------
        n_kx
            n_kx
        """

        raw_text = re.findall(r"generate\s*k-points\s*for:\s*(.*)", self.file_str)
        n_kx = None
        n_ky = None
        n_kz = None
        if len(raw_text) > 0:
            raw_text = raw_text[-1]
            n_kx = int(raw_text.split()[0])
            n_ky = int(raw_text.split()[1])
            n_kz = int(raw_text.split()[2])
        return n_kx, n_ky, n_kz

    @cached_property
    def kgrid(self):
        return self._get_axes_nk()

    @cached_property
    def n_kx(self):
        return self.kgrid[0]

    @cached_property
    def n_ky(self):
        return self.kgrid[1]

    @cached_property
    def n_kz(self):
        return self.kgrid[2]

    @cached_property
    def version(self):
        """
        Returns the version of the OUTCAR file.
        """
        return re.findall(r"vasp\.\d+\.\d+\.\d+", self.file_str)[-1].strip("vasp.")

    @cached_property
    def version_tuple(self):
        """
        Returns the version of the OUTCAR file as a tuple.
        """
        return tuple(int(x) for x in self.version.split("."))

    @cached_property
    def _fermi_values(self) -> list[str]:
        return re.findall(r"E-fermi\s*:\s*(-?\d+.\d+)", self.file_str)

    @property
    def has_fermi(self) -> bool:
        """Whether VASP wrote an "E-fermi :" line; a run killed early leaves none."""
        return bool(self._fermi_values)

    @cached_property
    def fermi(self) -> float:
        """
        Just finds all E-fermi fields in the outcar file and keeps the
        last one (if more than one found).

        Returns
        -------
        fermi
            the fermi energy
        """
        values = self._fermi_values
        if not values:
            raise ValueError(
                f"The OUTCAR {self.filepath or 'text'} has no 'E-fermi :' line, so it gives no"
                + " Fermi energy; the run stopped before VASP wrote one."
            )
        return float(values[-1])

    @cached_property
    def is_non_collinear(self) -> bool | None:
        """Whether the run was non-collinear, from the parameters VASP lists after
        "Startparameter for this run:", or None when the OUTCAR stops before them.

        The INCAR echo at the top of the OUTCAR is what the user wrote, not what VASP
        ran with: LSORBIT turns LNONCOLLINEAR on whatever the INCAR says.
        """
        start = self.file_str.find(_PARAMETER_LISTING)
        if start == -1:
            return None
        listed: dict[str, bool] = {}
        for match in _LISTED_FLAG.finditer(self.file_str, start):
            listed.setdefault(match.group(1).upper(), match.group(2).upper() in ("T", ".TRUE."))
            if len(listed) == 2:
                break
        if not listed:
            return None
        return listed.get("LNONCOLLINEAR", False) or listed.get("LSORBIT", False)

    @cached_property
    def reciprocal_lattice(self) -> np.ndarray:
        """
        Finds and return the reciprocal lattice vectors, if more than
        one set present, it return just the last one.

        Returns
        -------
        np.ndaaray
            return the reciprocal lattice vectors
        """

        start = self.file_str.rfind("reciprocal lattice vectors")
        if start == -1:
            raise ValueError("No reciprocal lattice vectors found")
        match = re.compile(r"reciprocal lattice vectors[\s\S]+?(?=\n\s?\n\s?)").search(
            self.file_str, start
        )
        assert match is not None
        numbers = re.findall(r"[-]?\d+\.\d+", match.group(0))

        # Create a NumPy array from the found numbers, reshape it to 3 rows and 6 columns
        all_vectors = np.array(numbers, dtype=float).reshape(3, 6)

        # Slice the array to get the last 3 columns,
        reciprocal_lattice = all_vectors[:, 3:]
        return reciprocal_lattice

    @cached_property
    def rotations(self) -> np.ndarray:
        """The operations VASP reduced the k-mesh with, as rotations of fractional k
        (k' = R k), shape (n, 3, 3).

        Read from the operator table when the OUTCAR prints one (VASP 5.4 "Space group
        operators", VASP 6 "irot" blocks at NWRITE = 3). Otherwise recovered: the one subgroup
        of the crystal's operations whose point group, size and orbits on the mesh match what
        the OUTCAR states, with ties between subgroups broken by the initial magnetic moments.
        Empty when the OUTCAR shows no reduced mesh, or when not exactly one subgroup remains.
        """
        sym_ops = self.get_symmetry_operations()
        if sym_ops:
            return np.array([sym_op["rotation"] for sym_op in sym_ops])

        # Every fact the fit reads is printed before the first electronic step.
        first_step = re.search(r"-{5,} Iteration", self.file_str)
        header = self.file_str[: first_step.start()] if first_step else self.file_str
        reduction = _mesh_reduction(header, self.kgrid, self.is_non_collinear)
        if reduction is None:
            return np.empty((0, 3, 3))
        fitting = _fitting_operations(reduction)
        if len(fitting) != 1:
            logger.info(
                "%d sets of the crystal's operations fit the reduced k-mesh in the OUTCAR",
                len(fitting),
            )
            return np.empty((0, 3, 3))
        return fitting[0].astype(float)

    def get_symmetry_operations(self) -> list[dict[str, Any]]:
        raw_spg_ops = re.search(r"Found\s+(\d+)\s+space group operations", self.file_str)

        if raw_spg_ops is None:
            return []

        n_spg_operations = int(raw_spg_ops.group(1))

        logger.debug(f"n_spg_operations: {n_spg_operations}")

        vasp54_block_match = re.search(
            r"Space group operators:\s*\n"  # header line
            + r"([ \t]*irot[\s\S]+?)"  # from the column headers …
            + r"(?=\n\s?\n\s?)",  # … up to the next blank/non-indented line
            self.file_str,
            flags=re.IGNORECASE,
        )

        vasp60_block_pattern = re.compile(
            r"""
            irot\s*:\s*(?P<irot>\d+)          # 1) operator index
            .*?                               # skip down to isymop
            isymop:\s*
            (?P<r1>-?\d+\s+-?\d+\s+-?\d+)\s*  # 2) first row of 3 ints
            (?P<r2>-?\d+\s+-?\d+\s+-?\d+)\s*  # 3) second row
            (?P<r3>-?\d+\s+-?\d+\s+-?\d+)     # 4) third row
            .*?                               # skip to gtrans
            gtrans:\s*
            (?P<g1>-?\d+\.\d+)\s+
            (?P<g2>-?\d+\.\d+)\s+
            (?P<g3>-?\d+\.\d+)
            .*?                               # skip to ptrans
            ptrans:\s*
            (?P<p1>-?\d+\.\d+)\s+
            (?P<p2>-?\d+\.\d+)\s+
            (?P<p3>-?\d+\.\d+)
            .*?                               # skip to rotmap
            rotmap:\s*\n                      # the “rotmap:” line might stand alone
            (?P<rotmap>                       # now capture one or more “( a-> b )” entries
            (?:[ \t]*\(\s*\d+\s*->\s*\d+\s*\)\s*)+
            )
            (?=\n\n)                       # stop at double new line
        """,
            re.VERBOSE | re.DOTALL,
        )

        vasp60_block_matches = vasp60_block_pattern.findall(self.file_str)

        if vasp54_block_match:
            logger.info("Detected Space Group Operators in a format from VASP 5.4")
            spg_operators: list[dict[str, Any]] = []
            block = vasp54_block_match.group(1).rstrip()
            logger.debug(f"space group operators block: \n{block}")

            headers = None
            for i, line in enumerate(block.splitlines()):
                values = line.split()
                if i == 0:
                    headers = values
                    continue

                assert headers is not None

                spg_operator: dict[str, Any] = {}
                for ispg, value in enumerate(values):
                    column_name: str = headers[ispg]
                    value = int(value) if column_name == "irot" else float(value)
                    spg_operator[column_name] = value

                spg_operators.append(spg_operator)

            sym_ops: list[dict[str, Any]] = []
            for operator in spg_operators:
                sym_op: dict[str, Any] = {}
                irot = operator["irot"]
                det_A = operator["det(A)"]
                # convert alpha to radians
                alpha = np.pi * operator["alpha"] / 180.0
                # get rotation axis
                x = operator["n_x"]
                y = operator["n_y"]
                z = operator["n_z"]
                tau_x = operator["tau_x"]
                tau_y = operator["tau_y"]
                tau_z = operator["tau_z"]

                gtrans = np.array([tau_x, tau_y, tau_z])

                R = (
                    np.array(
                        [
                            [
                                np.cos(alpha) + x**2 * (1 - np.cos(alpha)),
                                x * y * (1 - np.cos(alpha)) - z * np.sin(alpha),
                                x * z * (1 - np.cos(alpha)) + y * np.sin(alpha),
                            ],
                            [
                                y * x * (1 - np.cos(alpha)) + z * np.sin(alpha),
                                np.cos(alpha) + y**2 * (1 - np.cos(alpha)),
                                y * z * (1 - np.cos(alpha)) - x * np.sin(alpha),
                            ],
                            [
                                z * x * (1 - np.cos(alpha)) - y * np.sin(alpha),
                                z * y * (1 - np.cos(alpha)) + x * np.sin(alpha),
                                np.cos(alpha) + z**2 * (1 - np.cos(alpha)),
                            ],
                        ]
                    )
                    * det_A
                )

                R = np.linalg.inv(self.reciprocal_lattice.T).dot(R).dot(self.reciprocal_lattice.T)
                R = np.round(R, decimals=3)

                sym_op["irot"] = irot
                sym_op["rotation"] = R
                sym_op["gtrans"] = gtrans
                sym_op["ptrans"] = None
                sym_op["rotmap"] = None

                sym_ops.append(sym_op)

            return sym_ops

        elif vasp60_block_matches:
            logger.info("Detected Space Group Operators in a format from VASP 6.*")
            sym_ops = []
            for vasp60_block_match in vasp60_block_matches:
                sym_op = {}
                irot = int(vasp60_block_match[0])
                r1 = np.array([int(val) for val in vasp60_block_match[1].split()])
                r2 = np.array([int(val) for val in vasp60_block_match[2].split()])
                r3 = np.array([int(val) for val in vasp60_block_match[3].split()])
                g1 = float(vasp60_block_match[4])
                g2 = float(vasp60_block_match[5])
                g3 = float(vasp60_block_match[6])
                gtrans = np.array([g1, g2, g3])
                p1 = float(vasp60_block_match[7])
                p2 = float(vasp60_block_match[8])
                p3 = float(vasp60_block_match[9])
                ptrans = np.array([p1, p2, p3])
                rotmap = vasp60_block_match[10].strip()

                rotation = np.array([r1, r2, r3])
                rotation = np.array(rotation)

                sym_op["irot"] = irot
                sym_op["rotation"] = rotation.T
                sym_op["gtrans"] = gtrans
                sym_op["ptrans"] = ptrans
                sym_op["rotmap"] = rotmap

                sym_ops.append(sym_op)

            return sym_ops

        else:
            logger.info("No space group operators block found")
            return []

    def to_dict(self) -> dict[str, Any]:
        tmp_dict: dict[str, Any] = {}
        symops = self.get_symmetry_operations()
        for symop in symops:
            for key in symop:
                if isinstance(symop[key], np.ndarray):
                    symop[key] = symop[key].tolist()
        tmp_dict["symops"] = symops
        tmp_dict["fermi"] = self.fermi
        tmp_dict["reciprocal_lattice"] = self.reciprocal_lattice.tolist()
        tmp_dict["rotations"] = self.rotations.tolist()
        return tmp_dict

    @override
    def __contains__(self, key: str) -> bool:  # ty: ignore[invalid-method-override]
        return key in self.__dict__

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return len(self.__dict__)
