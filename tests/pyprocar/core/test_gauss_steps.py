"""Lagrange-Gauss reduction of two lattice rows ends, and returns the two shortest vectors.

When u.v / u.u rounds to 1/2 plus one unit in the last place, a step by u and the step back
alternate forever. Each case runs under a timeout, so that hang fails instead of stalling CI.

Expected values come from a brute-force search over small integer combinations of the rows:
the shortest lattice vector, and the shortest one independent of it.
"""

import itertools

import numpy as np
import pytest

from pyprocar.core._periodic_grid import _gauss_steps, periodic_grid
from tests.pyprocar.core.test_basis_invariance import _grid_mesh
from tests.utils.lattices import hexagonal, reciprocal, shear

pytestmark = pytest.mark.timeout(10)

HALF_PLUS_ULP_ROWS = np.array(
    [
        [0.42206178877649486, -0.08409185852105734, -0.19239186055413598],
        [0.7636451999940663, -0.28142151784007724, 0.0656856490359661],
    ]
)
"""Rows on which the reduction alternated mu = +1 and mu = -1 without end."""


def _shortest_pair(rows: np.ndarray, reach: int = 6) -> np.ndarray:
    """Lengths of the shortest nonzero combination of ``rows`` with coefficients up to
    ``reach``, and of the shortest one not parallel to it."""
    coefficients = np.array(
        [c for c in itertools.product(range(-reach, reach + 1), repeat=2) if any(c)]
    )
    vectors = coefficients @ rows
    lengths = np.linalg.norm(vectors, axis=1)
    first = int(lengths.argmin())
    independent = np.abs(coefficients @ np.array([coefficients[first, 1], -coefficients[first, 0]]))
    return np.array([lengths[first], lengths[independent > 0].min()])


def _about_z(degrees: float) -> np.ndarray:
    turn = np.radians(degrees)
    cos, sin = np.cos(turn), np.sin(turn)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def test_gauss_steps_ends_when_the_projection_rounds_to_half_plus_an_ulp():
    steps = _gauss_steps(HALF_PLUS_ULP_ROWS)

    assert abs(round(np.linalg.det(steps))) == 1
    lengths = np.sort(np.linalg.norm(steps @ HALF_PLUS_ULP_ROWS, axis=1))
    np.testing.assert_allclose(lengths, _shortest_pair(HALF_PLUS_ULP_ROWS), rtol=1e-12)


@pytest.mark.parametrize("kgrid", [(12, 12, 1), (9, 9, 1)])
@pytest.mark.parametrize("degrees", [270.0, 275.0])
def test_periodic_grid_draws_a_turned_hexagonal_plane(degrees, kgrid):
    """A hexagonal cell with rows 60 degrees apart, turned about z, on a 2D grid."""
    cell = shear(1) @ hexagonal(3.567, 20.0) @ _about_z(degrees).T
    lattice = reciprocal(cell)

    grid = periodic_grid(_grid_mesh(lattice, kgrid))

    assert grid is not None
    drawn_steps = (grid.lattice / np.array(grid.n)[:, None])[:2]
    k_plane = (lattice / np.array(kgrid)[:, None])[:2]
    in_k_plane = drawn_steps @ np.linalg.pinv(k_plane)
    np.testing.assert_allclose(in_k_plane, np.rint(in_k_plane), rtol=0, atol=1e-9)
    assert abs(round(np.linalg.det(np.rint(in_k_plane)))) == 1
    lengths = np.sort(np.linalg.norm(drawn_steps, axis=1))
    np.testing.assert_allclose(lengths, _shortest_pair(k_plane), rtol=1e-9)
