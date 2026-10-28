from collections.abc import Sequence

import numpy as np

from pyprocar.core.structure import Structure


def translation_images(
    positions: np.ndarray,
    labels: Sequence[object],
    translations: np.ndarray,
    lattice: np.ndarray,
    tol: float,
    names: Sequence[str],
) -> np.ndarray:
    """``images[t, j]``: the site that ``translations[t]`` carries site ``j`` onto.

    The image has the same label and the position nearest the translated site among
    those within ``tol`` of it in every fractional coordinate. Positions and
    translations are fractional supercell coordinates; ``names`` describe the sites
    for the error raised when one has no image.
    """
    positions = np.asarray(positions, dtype=float)
    labels = np.asarray(labels)
    same_label = labels[:, None] == labels[None, :]
    images = np.empty((len(translations), len(positions)), dtype=int)
    for t, r in enumerate(np.asarray(translations, dtype=float)):
        offset = positions[None, :, :] - (positions[:, None, :] + r)
        offset -= np.round(offset)
        candidate = same_label & np.all(np.abs(offset) < tol, axis=-1)
        distance = np.where(candidate, np.linalg.norm(offset @ lattice, axis=-1), np.inf)
        lost = np.flatnonzero(~candidate.any(axis=1))
        images[t] = distance.argmin(axis=1)
        unreached = np.setdiff1d(np.arange(len(positions)), images[t])
        if lost.size or unreached.size:
            what = (
                f"{names[lost[0]]} has no image"
                if lost.size
                else f"{names[unreached[0]]} is the image of no atom"
            )
            shift = ", ".join(f"{x + 0.0:.3g}" for x in r)
            raise ValueError(
                "Unfolding needs every primitive-cell translation to carry the atoms one to one"
                + f" onto atoms, and {what} within tol_radius={tol} under the translation"
                + f" ({shift}) in fractional supercell coordinates. Check that"
                + " transformation_matrix turns the primitive cell into this supercell; a"
                + " vacancy or an interstitial breaks this, so it cannot be unfolded."
            )
    return images


class Unfolder:
    def __init__(
        self,
        ebs=None,
        transformation_matrix=np.diag([1, 1, 1]),
        structure=None,
        tol_radius=0.1,
    ):
        self.ebs = ebs
        self.trans_mat = transformation_matrix
        self.structure = structure
        self.eigenvectors = None
        self.cell = structure.lattice
        self.tol_radius = tol_radius
        self.trans_rs = None
        self.trans_indices = None

        self._prepare_unfold_basis()
        self._make_translate_maps()

    @property
    def nfold(self):
        N = np.linalg.det(self.trans_mat).round(2)
        if not N.is_integer() or (1 / N).is_integer():
            raise ValueError("This transfare with is not proper.")
            return None
        else:
            return int(N)

    def _prepare_unfold_basis(self):
        self.eigenvectors = np.zeros(
            shape=(
                self.ebs.n_kpoints,
                self.ebs.n_bands,
                self.ebs.n_spins,
                self.ebs.n_atoms * self.ebs.n_orbitals,
            ),
            dtype=np.complex128,
        )
        for ispin in range(self.ebs.n_spins):
            self.eigenvectors[:, :, ispin, :] = np.reshape(
                self.ebs.projected_phase[:, :, ispin, :, :],
                (
                    self.ebs.n_kpoints,
                    self.ebs.n_bands,
                    self.ebs.n_atoms * self.ebs.n_orbitals,
                ),
            )

        # norm the atomic-orbital axis
        norm = np.linalg.norm(self.eigenvectors, ord=2, axis=-1)
        self.eigenvectors /= norm[..., None]

    def _make_translate_maps(self):
        """Map each basis function to its image under every primitive-cell translation.

        ``trans_indices[t, j]`` is the basis function that ``trans_rs[t]`` carries
        basis function ``j`` onto, so ``T(r_t) psi = psi[trans_indices[t]]``.
        """
        a1 = Structure(atoms=["H"], fractional_coordinates=[[0, 0, 0]], lattice=np.diag([1, 1, 1]))
        rs = a1.transform(self.trans_mat).fractional_coordinates
        n_atoms, n_orbitals = self.ebs.n_atoms, self.ebs.n_orbitals
        atom_images = translation_images(
            self.structure.fractional_coordinates,
            np.zeros(n_atoms),
            rs,
            self.cell,
            self.tol_radius,
            names=[f"atom {i} ({atom})" for i, atom in enumerate(self.structure.atoms)],
        )
        self.trans_rs = rs
        self.trans_indices = (
            atom_images[:, :, None] * n_orbitals + np.arange(n_orbitals)
        ).reshape(len(rs), n_atoms * n_orbitals)

    def _get_weight(self, evec):
        r"""
        W= sum_1^N < evec| T(r_i)exp(-I (K+G) * r_i| evec>, here
        G=0. T(r_i)exp(-I K r_i)| evec> = evec[indices[i]]

                    N
                1  ---
         W_KJ = -  \                   -j(K+G).r_i
                N  /   <KJ|T(r_i)|KJ> e
                   ---
                   i=1
        """
        assert self.trans_indices is not None
        weight = 0j
        N = self.nfold
        for ind in self.trans_indices:
            weight += np.vdot(evec, evec[ind]) / N

        return weight.real

    @property
    def weights(self):
        """
        Get the weight for all the modes.
        """
        nqpts, nfreqs = self.eigenvectors.shape[0], self.eigenvectors.shape[1]
        weights = np.zeros([nqpts, nfreqs, self.ebs.n_spins])
        for ispin in range(self.ebs.n_spins):
            for iqpt in range(nqpts):
                for ifreq in range(nfreqs):
                    weights[iqpt, ifreq, ispin] = self._get_weight(
                        self.eigenvectors[iqpt, ifreq, ispin, :]
                    )

        return weights
