"""Defect position identification without prior knowledge."""
import logging
import warnings
from collections import namedtuple
from typing import List, Tuple

import numpy as np
from monty.json import MSONable
from numpy.typing import ArrayLike, NDArray
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Optional imports
try:
    from dscribe.descriptors import SOAP
except ImportError:
    SOAP = None


__author__ = "Jimmy Shen"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy Shen @jmmshn"
__date__ = "Jan 24, 2022"

_logger = logging.getLogger(__name__)
DUMMY_SPECIES = "Si"

SiteVec = namedtuple("SiteVec", ["species", "site", "vec"])
SiteGroup = namedtuple("SiteGroup", ["species", "similar_sites", "vec"])


class DefectSiteFinder(MSONable):
    """Find the location of a defect with no pior knowledge."""

    def __init__(self, symprec: float = 0.01, angle_tolerance: float = 5.0):
        """Configure the behavior of the defect site finder.

        Args:
            symprec (float): Symmetry tolerance parameter for SpacegroupAnalyzer
            angle_tolerance (float): Angle tolerance parameter for SpacegroupAnalyzer
        """
        if SOAP is None:
            raise ImportError(
                "dscribe is required to use DefectSiteFinder. Install with ``pip install dscribe``."
            )
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

    def get_defect_fpos(
        self, defect_structure: Structure, base_structure: Structure
    ) -> ArrayLike:
        """Get the position of a defect in the pristine structure.

        Args:
            defect_structure: Relaxed structure containing the defect
            base_structure: Structure for the pristine cell

        Returns:
            ArrayLike: Position of the defect in the pristine structure
            (in fractional coordinates)
        """
        if self._is_impurity(defect_structure, base_structure):
            return self.get_impurity_position(defect_structure, base_structure)
        else:
            return self.get_native_defect_position(defect_structure, base_structure)

    def _is_impurity(
        self, defect_structure: Structure, base_structure: Structure
    ) -> bool:
        """Check if the defect structure is an impurity.

        Args:
            defect_structure: Structure containing the defect
            base_structure: Structure for the pristine cell

        Returns:
            bool: True if the defect structure is an impurity
        """
        # check if the defect structure is an impurity
        base_species = {site.species_string for site in base_structure}
        defect_species = {site.species_string for site in defect_structure}
        return len(defect_species - base_species) > 0

    def get_native_defect_position(
        self, defect_structure: Structure, base_structure: Structure
    ) -> ArrayLike:
        """Get the position of a native defect in the defect structure.

        Args:
            defect_structure: Relaxed structure containing the defect
            base_structure: Pristine structure without the defect

        Returns:
            ArrayLike: Position of the defect in the defect structure
            (in fractional coordinates)
        """
        distored_sites, distortions = list(
            zip(*self.get_most_distorted_sites(defect_structure, base_structure))
        )
        positions = [defect_structure[isite].frac_coords for isite in distored_sites]
        return get_weighted_average_position(
            defect_structure.lattice, positions, distortions
        )

    def get_impurity_position(
        self, defect_structure: Structure, base_structure: Structure
    ):
        """Get the position of an impurity defect.

        Look at all sites with impurity atoms, and take the average of the positions of
        the sites.

        Args:
            defect_structure: Relaxed structure containing the defect
            base_structure: Pristine structure without the defect

        Returns:
            ArrayLike: Position of the defect in the defect structure
        """
        # get the pbc average position of all sites not in the base structure
        base_species = {site.species_string for site in base_structure}
        impurity_sites = [
            *filter(lambda x: x.species_string not in base_species, defect_structure)
        ]
        return get_weighted_average_position(
            defect_structure.lattice, [s.frac_coords for s in impurity_sites]
        )

    def get_most_distorted_sites(
        self, defect_structure: Structure, base_structure: Structure
    ) -> List[Tuple[int, float]]:
        """Identify the set of sites with the most deviation from the pristine.

        Performs the following steps:

        1. For each site in the defect structure, find the closest site in the
            pristine structure.
        2. Then, compute a distortion field based on SOAP vectors.
        3. Filter the most distorted sites:
            - sort largest to smallest distortion
            - look at the diff in the sorted list
            - use the biggest value drop as the cutoff

        Args:
            defect_structure: Relaxed structure containing the defect
            base_structure: Structure for the pristine cell

        Returns:
            List[Tuple[int, float]]: List of (site index, distortion) pairs
        """
        pristine_groups = get_site_groups(
            struct=base_structure,
            symprec=self.symprec,
            angle_tolerance=self.angle_tolerance,
        )
        defect_vecs = get_site_vecs(defect_structure)
        res = []
        for i, v in enumerate(defect_vecs):
            (
                best_m,
                best_s,
            ) = best_match(v, pristine_groups)
            if v.species != best_m.species:
                warnings.warn(
                    "The species of a site in the distorted structure is different "
                    "from the species of the closest pristine site."
                )

            res.append((i, np.abs(best_s - 1)))

        res.sort(key=lambda x: x[1], reverse=True)
        deviations = [r[1] for r in res]
        bound = _get_broundary(deviations)
        return res[:bound]


# %%
def get_site_groups(struct, symprec=0.01, angle_tolerance=5.0) -> List[SiteGroup]:
    """Group the sites in the structure by symmetry.

    Group the sites in the structure by symmetry and return a
    list of ``SiteGroup`` namedtuples.

    Args:
        struct: Structure object to be analyzed
        symprec: Symmetry precision passed to SpacegroupAnalyzer
        angle_tolerance: Angle tolerance passed to SpacegroupAnalyzer

    Returns:
        List[SiteGroup]: List of SiteGroup namedtuples representing groups of
        symmetrically equivalent sites

    """
    sa = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tolerance)
    sstruct = sa.get_symmetrized_structure()
    site_groups = []
    groups = sstruct.equivalent_indices
    soap_vec = get_soap_vec(struct)
    for g in groups:
        sg = SiteGroup(
            species=sstruct[g[0]].species_string, similar_sites=g, vec=soap_vec[g[0]]
        )
        site_groups.append(sg)
    return site_groups


def get_soap_vec(struct: Structure) -> NDArray:
    """Get the SOAP vector for each site in the structure.

    Args:
        struct: Structure object to compute the SOAP vector for

    Returns:
        NDArray: SOAP vector for each site in the structure,
            shape (n_sites, n_soap_features)
    """
    adaptor = AseAtomsAdaptor()
    species_ = [str(el) for el in struct.composition.elements]
    dummy_structure = struct.copy()
    for el in species_:
        dummy_structure.replace_species({str(el): DUMMY_SPECIES})
    soap_desc = SOAP(species=[DUMMY_SPECIES], rcut=5, nmax=8, lmax=6, periodic=True)
    vecs = soap_desc.create(adaptor.get_atoms(dummy_structure))
    return vecs


def get_site_vecs(struct: Structure):
    """Get the SiteVec representation of each site in the structure."""
    vecs = get_soap_vec(struct)
    site_vecs = []
    for i, site in enumerate(struct):
        site_vecs.append(SiteVec(species=site.species_string, site=site, vec=vecs[i]))
    return site_vecs


def cosine_similarity(vec1, vec2) -> float:
    """Cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def best_match(sv: SiteVec, sgs: List[SiteGroup]) -> Tuple[SiteGroup, float]:
    """Find the best match for a site in the defect structure.

    Args:
        sv: SiteVec namedtuples representing a site in the defect structure
        sgs: List of SiteGroup namedtuples representing groups of
        symmetrically equivalent sites in the pristine structure

    Returns:
        SiteGroup: The group that represents the best match for ``sv``
        float: The cosine similarity between ``sv`` and the best match

    """
    best_match = None
    best_similarity = -np.inf
    for sg in sgs:
        if sv.species != sg.species:
            continue
        csim = cosine_similarity(sv.vec, sg.vec)
        if csim > best_similarity:
            best_similarity = csim
            best_match = sg
    if best_match is None:
        raise ValueError("No matching species found.")
    return best_match, best_similarity


def _get_broundary(arr, n_max=16, n_skip=3):
    """Get the boundary index for the high-distortion indices.

    Assuming arr is sorted in reverse order,
    find the biggest value drop in arr[n_skip:n_max].
    """
    sub_arr = np.array(arr[n_skip:n_max])
    diffs = sub_arr[1:] - sub_arr[:-1]
    return np.argmin(diffs) + n_skip + 1


def get_weighted_average_position(lattice, frac_positions, weights=None) -> NDArray:
    """Get the weighted average position of a set of positions in frac coordinates.

    The algorithm starts at position with the highest weight, and gradually moves
    the average point by finding the closest image of each additional position to the
    average point. This can be used to find the center of mass of a group of sites in a
    molecule in CH3NH3PbI3 (Note: Since the average positions in periodic system is not
    unique, this algorithm only works if the collection of positions is significantly
    smaller than the unit cell.)

    Args:
        frac_positions (3xN array-like): The positions to average.
        weights (1xN array-like): The weights of the positions.

    Returns
    -------
        NDArray: (3x1 array): The weighted average position in fractional coordinates.
    """
    if weights is None:
        weights = [1.0] * len(frac_positions)
    if len(frac_positions) != len(weights):
        raise ValueError("The number of positions and weights must be the same.")

    # TODO: can be replaced with the zip(..., strict=True) syntax in Python 3.10
    pos_weights = list(zip(frac_positions, weights))
    pos_weights.sort(key=lambda x: x[1], reverse=True)

    # initial guess at the center with zero weight
    p_guess = np.ones(3) * 0.5
    w_sum = 0

    for p, w in pos_weights:
        _, jimage = lattice.get_distance_and_image(p_guess, p)
        p_guess = (w_sum * p_guess + w * (p + jimage)) / (w_sum + w)
        w_sum += w
    return p_guess
