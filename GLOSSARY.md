# pyprocar

pyprocar reads electronic structure from DFT codes and plots bands, densities of states and Fermi surfaces. These terms cover the lattice geometry under the Fermi-surface plots, where a result must not depend on how the user wrote the cell.

## Language

### Lattices and bases

**Reciprocal lattice**:
The lattice spanned by b1, b2, b3, where a_i . b_j = delta_ij, in 1/Angstrom without 2 pi. With B the matrix of rows b1, b2, b3, the Cartesian k of fractional coordinates `frac` is `frac @ B`.
_Avoid_: reciprocal cell, k-space lattice

**Successive minima**:
The lengths of the shortest three linearly independent vectors of a lattice. They depend only on the lattice.

**Reduced basis**:
A basis whose sorted row lengths equal the lattice's successive minima, within tolerance. A symmetric lattice has several that are not reorderings or negations of each other.
_Avoid_: canonical basis, Niggli basis (for the general idea)

**Obtuse superbase**:
A basis of the lattice together with minus the sum of its rows, where all four vectors meet pairwise at obtuse angles. Any three of them form a Selling-reduced basis.
_Avoid_: Selling basis

**Basis variant**:
One way to give the same lattice: a relabelling, a shear, or both.

**Relabelling**:
The same rows reordered, negated, turned, or changed by rounding noise. Each lattice vector keeps the same indices up to sign and order.
_Avoid_: permutation (alone)

**Shear**:
An integer unimodular change of basis that is not a signed permutation. It gives the same lattice, often in a basis that is not reduced.

### Drawing a Fermi surface

**K-point lattice**:
The lattice spanned by the reciprocal lattice rows, each divided by the grid size along its axis. The points of a uniform k-grid are this lattice, or a translate of it for a shifted grid.
_Avoid_: k-mesh lattice

**Drawn basis**:
The basis of the k-point lattice whose steps index the drawn grid.

**Drawn grid**:
The k-point lattice indexed in the drawn basis over the box around the first Brillouin zone. It holds copies of the stored samples, and no value is interpolated.
_Avoid_: reduced grid, padded grid

**Label independence**:
A result is label-independent when every relabelling of the given reciprocal rows gives the same result, turned with the cell.

**Basis invariance**:
A result is basis-invariant when every basis variant gives the same result, turned with the cell, so it depends on the lattice only.
_Avoid_: label independence (for this stronger property)

## Relationships

- A **basis variant** is a **relabelling**, a **shear**, or both.
- A **drawn grid** samples one **k-point lattice** in one **drawn basis**.
- The **drawn basis** is a **reduced basis** of the **k-point lattice**, unless the given reciprocal basis is reduced or an **obtuse superbase**.
- **Basis invariance** implies **label independence**, but not the reverse.

## Flagged ambiguities

- "Canonical basis" (issue #59) meant one basis that every caller uses and that depends on the lattice only. No such **reduced basis** exists when the lattice's symmetry permutes its reduced bases, as for fcc and rhombohedral cells. Resolved: say **reduced basis**, and state whether a result needs **label independence** or **basis invariance**.
- "Reduced grid" (issue #59) meant a candidate **drawn grid** in one of several **reduced bases**. Resolved: say **drawn grid** for the one drawn, and "the drawn grid in each reduced basis" for the candidates.
