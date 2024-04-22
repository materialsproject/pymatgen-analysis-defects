"""Functions for creating supercells for defect calculations."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
from monty.dev import deprecated
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import Lattice
from pymatgen.util.coord_cython import pbc_shortest_vectors

# from ase.build import find_optimal_cell_shape, get_deviation_from_optimal_cell_shape
# from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from pymatgen.core import Structure

__author__ = "Jimmy-Xuan Shen"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy-Xuan Shen"
__email__ = "jmmshn@gmail.com"

_logger = logging.getLogger(__name__)


def get_sc_fromstruct(
    base_struct: Structure,
    min_atoms: int = 80,
    max_atoms: int = 240,
    min_length: float = 10.0,
    force_diagonal: bool = False,
) -> NDArray | ArrayLike | None:
    """Generate the best supercell from a unitcell.

    The CubicSupercellTransformation from PMG is much faster but don't iterate over as
    many supercell configurations so it's less able to find the best configuration in a
    given cell size. We try the PMG's cubic supercell transformation first and if it fails
    we will use the `find_optimal_cell_shape` function from ASE which is much slower but
    exhaustive.

    Args:
        base_struct: structure of the unit cell
        max_atoms: Maximum number of atoms allowed in the supercell.
        min_atoms: Minimum number of atoms allowed in the supercell.
        min_length: Minimum length of the smallest supercell lattice vector.
        force_diagonal: If True, return a transformation with a diagonal transformation matrix.

    Returns:
        struc_sc: Supercell that is as close to cubic as possible
    """
    return _cubic_cell(
        base_struct,
        min_atoms,
        max_atoms=max_atoms,
        min_length=min_length,
        force_diagonal=force_diagonal,
    )


def get_matched_structure_mapping_old(
    uc_struct: Structure,
    sc_struct: Structure,
    sm: StructureMatcher | None = None,
) -> tuple[NDArray, ArrayLike] | None:  # pragma: no cover
    """Get the mapping of the supercell to the unit cell.

    Get the mapping from the supercell structure onto the base structure,
    Note: this only works for structures that are exactly matched.

    Args:
        uc_struct: host structure, smaller cell
        sc_struct: bigger cell
        sm: StructureMatcher instance
    Returns:
        sc_m : supercell matrix to apply to s1 to get s2
        total_t : translation to apply on s1 * sc_m to get s2
    """
    if sm is None:
        sm = StructureMatcher(primitive_cell=False, comparator=ElementComparator())
    s1, s2 = sm._process_species([uc_struct, sc_struct])
    fu, _ = sm._get_supercell_size(s1, s2)
    try:
        val, dist, sc_m, total_t, mapping = sm._strict_match(
            s1,
            s2,
            fu=fu,
            s1_supercell=True,
        )
    except TypeError:
        return None
    return sc_m, total_t


@deprecated(message="This function was reworked in Feb 2024")
def get_matched_structure_mapping(
    uc_struct: Structure,
    sc_struct: Structure,
    sm: StructureMatcher | None = None,
) -> tuple[NDArray, ArrayLike] | None:
    """Get the mapping of the supercell to the unit cell.

    Get the mapping from the supercell structure onto the base structure,
    Note: this only works for structures that are exactly matched.

    Args:
        uc_struct: host structure, smaller cell
        sc_struct: bigger cell
        sm: StructureMatcher instance
    Returns:
        sc_m : supercell matrix to apply to s1 to get s2
        total_t : translation to apply on s1 * sc_m to get s2
    """
    if sm is None:
        sm = StructureMatcher(
            primitive_cell=False,
            comparator=ElementComparator(),
            attempt_supercell=True,
        )
    s1, s2 = sm._process_species([sc_struct.copy(), uc_struct.copy()])
    trans = sm.get_transformation(s1, s2)
    if trans is None:
        return None
    sc, t, mapping = trans
    temp = s2.copy().make_supercell(sc)
    ii, jj = 0, mapping[0]
    vec = np.round(sc_struct[ii].frac_coords - temp[jj].frac_coords)
    return sc, t + vec


def _cubic_cell(
    base_struct: Structure,
    min_atoms: int = 80,
    max_atoms: int = 240,
    min_length: float = 10.0,
    force_diagonal: bool = False,
) -> NDArray | None:
    """Generate the best supercell from a unit cell.

    This is done using the pymatgen CubicSupercellTransformation class.

    Args:
        base_struct: structure of the unit cell
        max_atoms: Maximum number of atoms allowed in the supercell.
        min_atoms: Minimum number of atoms allowed in the supercell.
        min_length: Minimum length of the smallest supercell lattice vector.
        force_diagonal: If True, return a transformation with a diagonal transformation matrix.

    Returns:
        3x3 matrix: supercell matrix
    """
    from pymatgen.transformations.advanced_transformations import (
        CubicSupercellTransformation,
    )

    cst = CubicSupercellTransformation(
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        min_length=min_length,
        force_diagonal=force_diagonal,
    )

    try:
        cst.apply_transformation(base_struct)
    except AttributeError:
        return _ase_cubic(
            base_struct,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=min_length,
        )
    return cst.transformation_matrix


def _ase_cubic(
    base_structure: Structure,
    min_atoms: int = 80,
    max_atoms: int = 240,
    min_length: float = 10.0,
) -> NDArray:
    """Generate the best supercell from a unit cell.

    Use ASE's find_optimal_cell_shape function to find the best supercell.

    Args:
        base_structure: structure of the unit cell
        max_atoms: Maximum number of atoms allowed in the supercell.
        min_atoms: Minimum number of atoms allowed in the supercell.
        min_length: Minimum length of the smallest supercell lattice vector.

    Returns:
        3x3 matrix: supercell matrix
    """
    from ase.build import find_optimal_cell_shape, get_deviation_from_optimal_cell_shape
    from pymatgen.io.ase import AseAtomsAdaptor

    _logger.warning("ASE cubic supercell generation.")

    aaa = AseAtomsAdaptor()
    ase_atoms = aaa.get_atoms(base_structure)
    lower = math.ceil(min_atoms / base_structure.num_sites)
    upper = math.floor(max_atoms / base_structure.num_sites)
    min_dev = (float("inf"), None)
    for size in range(lower, upper + 1):
        _logger.warning("Trying size  %s", size)
        sc = find_optimal_cell_shape(
            ase_atoms.cell,
            target_size=size,
            target_shape="sc",
        )
        sc_cell = aaa.get_atoms(base_structure * sc).cell
        lattice_lens = np.linalg.norm(sc_cell, axis=1)
        if min(lattice_lens) < min_length:
            continue
        deviation = get_deviation_from_optimal_cell_shape(sc_cell, target_shape="sc")
        min_dev = min(min_dev, (deviation, sc))
    if min_dev[1] is None:
        msg = "Could not find a cubic supercell"
        raise RuntimeError(msg)
    return min_dev[1]


def _avg_lat(l1: Lattice, l2: Lattice) -> Lattice:
    """Get the average lattice from two lattices."""
    params = (np.array(l1.parameters) + np.array(l2.parameters)) / 2
    return Lattice.from_parameters(*params)


def _lowest_dist(struct: Structure, ref_struct: Structure) -> ArrayLike:
    """For each site, return the lowest distance to any site in the reference structure."""
    avg_lat = _avg_lat(struct.lattice, ref_struct.lattice)
    _, d_2 = pbc_shortest_vectors(
        avg_lat,
        struct.frac_coords,
        ref_struct.frac_coords,
        return_d2=True,
    )
    return np.min(d_2, axis=1)


def get_closest_sc_mat(
    uc_struct: Structure,
    sc_struct: Structure,
    sm: StructureMatcher | None = None,
    debug: bool = False,
) -> NDArray:
    """Get the best guess for the supercell matrix that created this defect cell.

    Args:
        uc_struct: unit cell structure, should be the host structure
        sc_struct: supercell structure, should be the defect structure
        sm: StructureMatcher instance, if None, one will be created with default settings
        debug: bool, if True, return the full list of (distances, lattice, sc_mat) will
            be returned

    Returns:
        sc_mat: supercell matrix to apply to s1 to get s2
        dist: mean distance between the two structures
    """
    if sm is None:
        sm = StructureMatcher(primitive_cell=False, comparator=ElementComparator())

    fu = int(np.round(sc_struct.lattice.volume / uc_struct.lattice.volume))
    candidate_lattices = tuple(
        sm._get_lattices(sc_struct.lattice, uc_struct, supercell_size=fu),
    )

    def _get_mean_dist(lattice: Lattice, sc_mat: NDArray) -> float:
        if (
            np.dot(np.cross(lattice.matrix[0], lattice.matrix[1]), lattice.matrix[2])
            < 0
        ):
            return float("inf")
        sc2 = uc_struct * sc_mat
        return np.mean(_lowest_dist(sc2, sc_struct))

    _, best_sc_mat = min(candidate_lattices, key=lambda x: _get_mean_dist(x[0], x[1]))
    if debug:
        return sorted(
            [
                (_get_mean_dist(lat_, smat_), lat_, smat_)
                for lat_, smat_ in candidate_lattices
            ],
            key=lambda x: x[0],
        )
    return best_sc_mat
