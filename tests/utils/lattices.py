"""The crystal lattices geometry tests run on, and the changes of basis that must not matter.

A real-space cell is a 3x3 array whose rows are the lattice vectors a1, a2, a3 in Angstrom.
``reciprocal(real)`` gives its reciprocal lattice the way the core holds it: rows b1, b2, b3 in
1/Angstrom without 2 pi, so the Cartesian k of fractional ``frac`` is ``frac @ reciprocal(real)``.

A result that depends only on the lattice must not change over ``basis_variants(cell)``, for every
cell in ``STANDARD_CELLS``: the cells that broke basis-dependent code before (#302) are the fcc
primitive cell, the acute Bi2Se3 rhombohedral cell, a row permutation, a sign flip and 1e-16 of
rounding noise.
"""

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

_HEX_ROWS = np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.0]])


def reciprocal(real: np.ndarray) -> np.ndarray:
    """Rows b_i with a_i . b_j = delta_ij: 1/Angstrom, without 2 pi."""
    return np.linalg.inv(real).T


def cubic(a: float) -> np.ndarray:
    return a * np.eye(3)


def fcc_primitive(a: float) -> np.ndarray:
    """The primitive rows (0 1 1) a/2, (1 0 1) a/2, (1 1 0) a/2 of the fcc lattice with cubic
    edge a. Its reciprocal lattice is ``bcc_primitive(2 / a)``, up to rounding."""
    return a / 2 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


def bcc_primitive(a: float) -> np.ndarray:
    """The primitive rows (-1 1 1) a/2, (1 -1 1) a/2, (1 1 -1) a/2 of the bcc lattice with cubic
    edge a. Its reciprocal lattice is ``fcc_primitive(2 / a)``, up to rounding."""
    return bct_primitive(a, a)


def bct_primitive(a: float, c: float) -> np.ndarray:
    """The primitive rows (-a a c)/2, (a -a c)/2, (a a -c)/2 of the body-centred tetragonal
    lattice with edges a, a and c; ``bcc_primitive(a)`` when c = a."""
    return 0.5 * np.array([[-a, a, c], [a, -a, c], [a, a, -c]])


def hexagonal(a: float, c: float) -> np.ndarray:
    """Rows a1 = (a, 0, 0) and a2 at 120 degrees to it, and a3 = (0, 0, c)."""
    return _HEX_ROWS * [a, a, c]


def rhombohedral(a: float, alpha: float) -> np.ndarray:
    """The rhombohedral primitive cell with edge a and angle alpha (degrees), its threefold axis
    along z."""
    cos = math.cos(math.radians(alpha))
    tx, ty, tz = math.sqrt((1 - cos) / 2), math.sqrt((1 - cos) / 6), math.sqrt((1 + 2 * cos) / 3)
    return a * np.array([[tx, -ty, tz], [0, 2 * ty, tz], [-tx, -ty, tz]])


def shear(k: int) -> np.ndarray:
    """The change of basis r1, r2 + k r1, r3: ``shear(k) @ rows`` is the same lattice."""
    return np.array([[1, 0, 0], [k, 1, 0], [0, 0, 1]])


AU_VAN_ALPHEN = 2.949546 * np.array(
    [[1.0, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)]]
)
"""Au fcc as the van-alphen fixture's POSCAR gives it: primitive rows 60 degrees apart."""

BI2SE3_RHOMBOHEDRAL = np.array(
    [
        [2.068734, -1.194384, 9.546678],
        [0.0, 2.388768, 9.546678],
        [-2.068734, -1.194384, 9.546678],
    ]
)
"""The rhombohedral primitive cell of Bi2Se3 (a = 9.841 Angstrom, alpha = 24.27 degrees): not
reduced, its successive minima are 4.14, 4.14 and 9.84 Angstrom."""

OBTUSE_RHOMBOHEDRAL = rhombohedral(5.0, 110.0)
"""Not reduced, and several reduced bases are equally near it in index space."""

MONOCLINIC = np.array([[1.0, 0.0, 0.0], [0.0, 1.3, 0.0], [0.4, 0.0, 0.9]])
TRICLINIC = np.array([[1.0, 0.1, 0.05], [0.3, 1.2, 0.0], [0.2, 0.35, 0.8]])

STANDARD_CELLS: dict[str, np.ndarray] = {
    "SrVO3 cubic": cubic(3.84652),
    "Au fcc, van-alphen POSCAR": AU_VAN_ALPHEN,
    "fcc primitive": fcc_primitive(4.08),
    "bcc primitive": bcc_primitive(3.3),
    "hexagonal c/a 2.72": hexagonal(2.46, 6.7),
    "hexagonal c/a 1.62": hexagonal(3.21, 5.21),
    "Bi2Se3 rhombohedral": BI2SE3_RHOMBOHEDRAL,
    "obtuse rhombohedral": OBTUSE_RHOMBOHEDRAL,
    "monoclinic": MONOCLINIC,
    "triclinic": TRICLINIC,
}
"""Real-space rows of cells, as their POSCARs give them."""

SHEARS: dict[str, np.ndarray] = {
    "r2 + 3 r1": shear(3),
    "r3 + 6 r1": np.array([[1, 0, 0], [0, 1, 0], [6, 0, 1]]),
    "r2 + 3 r1, r3 + 3 r1 + 3 r2": np.array([[1, 0, 0], [3, 1, 0], [3, 3, 1]]),
    "r2 + 3 r1, r3 - 2 r1 + 2 r2": np.array([[1, 0, 0], [3, 1, 0], [-2, 2, 1]]),
    "r2 - 2 r1, r3 + r1 + 2 r2": np.array([[1, 0, 0], [-2, 1, 0], [1, 2, 1]]),
    "U1": np.array([[0, -1, 4], [2, -2, -5], [1, -2, 2]]),
    "r1 + r3, r2 + r3": np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]),
}
"""Unimodular changes of basis other than a signed permutation: ``shear @ rows`` is the same
lattice in a basis that is not reduced."""

TURN = np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]])
"""A rotation (orthonormal rows, det 1): ``rows @ TURN.T`` is the cell a POSCAR in another
orientation gives, and its reciprocal lattice carries 1e-16 of rounding."""

_NOISE = np.array([[1, -1, 1], [-1, 1, 1], [1, 1, -1]]) * np.finfo(np.float64).eps
"""Relative changes of one unit in the last place, in a fixed pattern of signs."""


@dataclass(frozen=True, eq=False)
class BasisVariant:
    """One way to give a cell: ``real`` is ``change @ cell @ turn.T``, exactly for a relabelling,
    a shear and a turn, and within 1e-16 relative for rounding noise."""

    name: str
    real: np.ndarray
    change: np.ndarray
    """Integer unimodular: row i of ``real`` is sum_j change[i, j] a_j of the cell, turned."""
    turn: np.ndarray

    @property
    def is_relabelling(self) -> bool:
        """The rows are the cell's rows, reordered, negated, turned or rounded: their indices
        up to n are the cell's indices up to n."""
        return bool((np.abs(self.change).sum(axis=0) == 1).all())

    @property
    def reciprocal(self) -> np.ndarray:
        return reciprocal(self.real)


def basis_variants(cell: np.ndarray) -> Iterator[BasisVariant]:
    """The cell as given, sheared by each of ``SHEARS``, with its rows in every other order, with
    one row negated, turned by ``TURN``, sheared by U1 and turned, and with 1e-16 of rounding noise
    in every entry."""
    identity = np.eye(3, dtype=int)
    yield BasisVariant("given", cell, identity, np.eye(3))
    for name, change in SHEARS.items():
        yield BasisVariant(name, change @ cell, change, np.eye(3))
    for order in itertools.permutations(range(3)):
        if order != (0, 1, 2):
            change = identity[list(order)]
            name = "rows " + " ".join(f"r{i + 1}" for i in order)
            yield BasisVariant(name, change @ cell, change, np.eye(3))
    for row in range(3):
        change = identity.copy()
        change[row, row] = -1
        yield BasisVariant(f"-r{row + 1}", change @ cell, change, np.eye(3))
    yield BasisVariant("turned", cell @ TURN.T, identity, TURN)
    yield BasisVariant("U1, turned", SHEARS["U1"] @ cell @ TURN.T, SHEARS["U1"], TURN)
    yield BasisVariant("rounding noise", cell * (1 + _NOISE), identity, np.eye(3))
