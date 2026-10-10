"""Results that depend only on the lattice do not depend on the basis it is given in (#302).

Every case runs on every cell of ``STANDARD_CELLS`` in every one of its ``basis_variants``. A
relabelling (rows reordered, a row negated, the cell turned, 1e-16 of rounding) lists the same
lattice vectors with the same indices up to n, so it must give the same drawn k-grid and snap the
same directions. A shear gives the lattice in a basis that is not reduced, so it must still snap
the directions with indices up to 2 in the cell's own rows. Every variant builds the same zone.

Expected values come from the cell itself: the exact direction ``uvw @ cell``, the k-point lattice
the grid samples, and the Voronoi cell of the reciprocal lattice from scipy.
"""

import itertools
from functools import cache

import numpy as np
import pytest
from scipy.spatial import ConvexHull, Voronoi

from pyprocar.core._periodic_grid import periodic_grid
from pyprocar.core.brillouin_zone import BrillouinZone, zone_face_steps
from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo
from pyprocar.plotter._surface_plot import snap_normal
from tests.utils.lattices import STANDARD_CELLS, BasisVariant, basis_variants, reciprocal

REASON = (
    "dev passes this contract; it keeps #302's regressions out: fcc [113] and [223] at 88c0b927,"
    + " Bi2Se3 directions and a row-order dependent drawn basis at e86ee210"
)
VARIANTS = [
    (cell_name, variant)
    for cell_name, cell in STANDARD_CELLS.items()
    for variant in basis_variants(cell)
]
CASES = [pytest.param(name, variant, id=f"{name}-{variant.name}") for name, variant in VARIANTS]
SHEARED = [
    case for case, (_, variant) in zip(CASES, VARIANTS, strict=True) if not variant.is_relabelling
]


def _primitive_indices(reach: int) -> list[tuple[int, int, int]]:
    return [
        (u, v, w)
        for u, v, w in itertools.product(range(-reach, reach + 1), repeat=3)
        if (u, v, w) != (0, 0, 0) and np.gcd.reduce(np.abs([u, v, w])) == 1
    ]


LOW_INDICES = [*_primitive_indices(2), (3, 3, -1), (3, 3, 1), (-1, 3, 3), (1, 3, 3)]
"""Indices up to 2 in the cell's rows, and cubic [1 1 3] and [2 2 3] on the fcc primitive
rows (on other cells, four more directions of index 3)."""
GIVEN_INDICES = _primitive_indices(4)
"""The candidates before #302: every direction with indices up to 4 in the given rows."""
RELABELLED = [
    case for case, (_, variant) in zip(CASES, VARIANTS, strict=True) if variant.is_relabelling
]


UNSNAPPED = np.random.default_rng(302).normal(size=(3, 3))
"""Normals far from every low-index direction, which must stay as given."""


def _snap_misses(
    variant: BasisVariant, cell: np.ndarray, indices: list[tuple[int, int, int]]
) -> list[tuple[int, ...]]:
    """The directions ``uvw @ cell``, typed to 4 digits, that do not snap back to the exact
    direction in ``variant``, or snap under indices that are not that direction in its rows,
    and the unsnapped normals that snap."""
    missed: list[tuple[int, ...]] = []
    for uvw in indices:
        exact = np.array(uvw) @ cell @ variant.turn.T
        exact /= np.linalg.norm(exact)
        direction, named = snap_normal(np.round(exact, 4), variant.reciprocal)
        if not np.allclose(direction, exact, rtol=0, atol=1e-12):
            missed.append(uvw)
        # None: the rounded normal was already the exact direction, as [2 1 0] is on cubic.
        elif named is not None:
            along = np.array(named) @ variant.real
            if not all(type(i) is int for i in named) or not np.allclose(
                along / np.linalg.norm(along), exact, rtol=0, atol=1e-12
            ):
                missed.append(uvw)
    for normal in UNSNAPPED @ variant.turn.T:
        if snap_normal(normal, variant.reciprocal)[1] is not None:
            missed.append(tuple(np.round(normal, 3)))
    return missed


@pytest.mark.guards_existing_behaviour(reason=REASON)
@pytest.mark.parametrize(("cell_name", "variant"), RELABELLED)
def test_snap_normal_snaps_every_direction_with_given_indices_up_to_4_in_any_relabelling(
    cell_name, variant
):
    """Before #302 the candidates were the directions with indices up to 4 in the given rows. A
    relabelling of the rows has the same directions with indices up to 4, so each snaps, to the
    exact direction, under integer indices in the rows given."""
    cell = STANDARD_CELLS[cell_name]

    missed = _snap_misses(variant, cell, GIVEN_INDICES)

    assert len(GIVEN_INDICES) == 578
    assert missed == []


@pytest.mark.guards_existing_behaviour(reason=REASON)
@pytest.mark.parametrize(("cell_name", "variant"), SHEARED)
def test_snap_normal_snaps_the_low_index_directions_in_any_basis(cell_name, variant):
    """A shear does not change the lattice, so the directions with indices up to 2 in the cell's
    rows, and fcc [1 1 3] and [2 2 3], still snap."""
    cell = STANDARD_CELLS[cell_name]

    missed = _snap_misses(variant, cell, LOW_INDICES)

    assert len(LOW_INDICES) == 102
    assert missed == []


def _grid_mesh(reciprocal_lattice: np.ndarray, kgrid: tuple[int, int, int]):
    axes = [np.arange(n) / n for n in kgrid]
    frac = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(kgrid=kgrid, kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.0, 0.0, 0.0)),
        kpoints=frac,
        bands=np.zeros((len(frac), 1, 1)),
        projected=np.ones((len(frac), 1, 1, 1, 1)),
        fermi=0.0,
        reciprocal_lattice=reciprocal_lattice,
        orbital_names=["s"],
    )


def _assert_same_points(found: np.ndarray, expected: np.ndarray, atol: float) -> None:
    """The two sets of points match one to one within ``atol``."""
    distance = np.linalg.norm(found[:, None, :] - expected[None, :, :], axis=2)
    assert len(found) == len(expected)
    assert distance.min(axis=0).max() <= atol
    assert distance.min(axis=1).max() <= atol


def _drawn_grid_cases() -> list:
    cases = []
    for kgrid, grid_id in [((6, 6, 6), "6x6x6"), ((4, 6, 8), "4x6x8")]:
        for name, variant in VARIANTS:
            if not variant.is_relabelling:
                continue
            case_id = f"{name}-{variant.name}-{grid_id}"
            cases.append(pytest.param(name, variant, kgrid, id=case_id))
    return cases


@pytest.mark.guards_existing_behaviour(reason=REASON)
@pytest.mark.parametrize(("cell_name", "variant", "kgrid"), _drawn_grid_cases())
def test_drawn_k_grid_does_not_depend_on_the_order_signs_or_rounding_of_the_rows(
    cell_name, variant, kgrid
):
    """The k-points of the cell's grid ``kgrid``, given in a relabelled basis, are the same
    points: the drawn grid steps (rows of ``lattice / n``) are the same vectors up to sign, as
    drawn from the cell as given. Each set of steps is a basis of the k-point lattice."""
    cell = STANDARD_CELLS[cell_name]
    k_lattice = reciprocal(cell) / np.array(kgrid)[:, None]
    n1, n2, n3 = (int(n) for n in np.abs(variant.change) @ np.array(kgrid))

    given = periodic_grid(_grid_mesh(reciprocal(cell), kgrid))
    found = periodic_grid(_grid_mesh(variant.reciprocal, (n1, n2, n3)))

    assert given is not None and found is not None
    given_steps = given.lattice / np.array(given.n)[:, None]
    found_steps = found.lattice / np.array(found.n)[:, None] @ variant.turn
    in_k_lattice = given_steps @ np.linalg.inv(k_lattice)
    np.testing.assert_allclose(in_k_lattice, np.rint(in_k_lattice), rtol=0, atol=1e-9)
    assert abs(np.linalg.det(np.rint(in_k_lattice))) == pytest.approx(1.0)
    _assert_same_points(
        np.vstack([found_steps, -found_steps]),
        np.vstack([given_steps, -given_steps]),
        atol=1e-9 * np.abs(k_lattice).max(),
    )


def _successive_minima(lattice: np.ndarray) -> np.ndarray:
    """Lengths of the shortest three independent vectors of the lattice with these rows, from
    every combination of the rows with coefficients up to 3."""
    combos = np.array(list(itertools.product(range(-3, 4), repeat=3)))
    vectors = combos[np.any(combos, axis=1)] @ lattice
    picked: list[np.ndarray] = []
    for vector in vectors[np.argsort(np.linalg.norm(vectors, axis=1), kind="stable")]:
        if np.linalg.matrix_rank(np.array([*picked, vector]), tol=1e-9) > len(picked):
            picked.append(vector)
    return np.linalg.norm(np.array(picked), axis=1)


@pytest.mark.guards_existing_behaviour(
    reason="dev draws every shear of these cells in a reduced basis; #59's tie rule must keep it"
)
@pytest.mark.parametrize("kgrid", [(6, 6, 6), (16, 16, 16)], ids=["6x6x6", "16x16x16"])
@pytest.mark.parametrize(("cell_name", "variant"), SHEARED)
def test_drawn_k_grid_of_a_sheared_basis_is_reduced(cell_name, variant, kgrid):
    """N points along each sheared row sample the cell's own N^3 k-point lattice, as the shear is
    unimodular. The drawn grid steps are as short as its successive minima, as they are for the
    cell's own rows when those are reduced: a shear must not stretch the cells marched on."""
    minima = _successive_minima(reciprocal(STANDARD_CELLS[cell_name]) / np.array(kgrid)[:, None])

    found = periodic_grid(_grid_mesh(variant.reciprocal, kgrid))

    assert found is not None
    steps = found.lattice / np.array(found.n)[:, None]
    # POSCAR rounding: the Bi2Se3 rows give lengths 8e-8 apart that are equal in the crystal.
    np.testing.assert_allclose(np.sort(np.linalg.norm(steps, axis=1)), minima, rtol=1e-4)


@cache
def _voronoi_zone(cell_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Face vectors and vertices of the Voronoi cell of the origin among the 9^3 block of
    reciprocal lattice points of the cell, from scipy.spatial.Voronoi."""
    lattice = reciprocal(STANDARD_CELLS[cell_name])
    steps = np.array(list(itertools.product(range(-4, 5), repeat=3)))
    origin = len(steps) // 2
    voronoi = Voronoi(steps @ lattice)
    pairs = voronoi.ridge_points[(voronoi.ridge_points == origin).any(axis=1)]
    neighbours = pairs.sum(axis=1) - origin
    # The block reaches beyond every face vector, so no face of the origin's cell is missing.
    assert np.abs(steps[neighbours]).max() < 4
    vertices = voronoi.vertices[voronoi.regions[voronoi.point_region[origin]]]
    return voronoi.points[neighbours], vertices


@pytest.mark.guards_existing_behaviour(reason=REASON)
@pytest.mark.parametrize(("cell_name", "variant"), CASES)
def test_brillouin_zone_is_the_voronoi_cell_in_any_basis(cell_name, variant):
    """The zone's face vectors and corners are those of the Voronoi cell of the reciprocal
    lattice, turned with the cell, and its volume is that of the reciprocal cell."""
    faces, vertices = _voronoi_zone(cell_name)
    turn = variant.turn
    basis = variant.reciprocal

    found_faces = zone_face_steps(basis) @ basis
    zone = BrillouinZone(basis)

    _assert_same_points(found_faces @ turn, faces, atol=1e-9)
    _assert_same_points(np.asarray(zone.points) @ turn, vertices, atol=1e-9)
    volume = abs(np.linalg.det(reciprocal(STANDARD_CELLS[cell_name])))
    assert ConvexHull(zone.points).volume == pytest.approx(volume, rel=1e-9)
