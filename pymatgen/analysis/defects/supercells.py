"""Functions for creating supercells for defect calculations."""

from __future__ import annotations

import logging
import math

import numpy as np
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher

# from ase.build import find_optimal_cell_shape, get_deviation_from_optimal_cell_shape
# from pymatgen.io.ase import AseAtomsAdaptor
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
) -> np.ndarray | np.array | None:
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
    sc_mat = _cubic_cell(
        base_struct,
        min_atoms,
        max_atoms=max_atoms,
        min_length=min_length,
        force_diagonal=force_diagonal,
    )
    return sc_mat


def get_matched_structure_mapping(
    uc_struct: Structure, sc_struct: Structure, sm: StructureMatcher | None = None
):
    """Get the mapping of the supercell to the unit cell.

    Get the mapping from the supercell defect structure onto the base structure,

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
            s1, s2, fu=fu, s1_supercell=True
        )
    except TypeError:
        return None
    return sc_m, total_t


def _cubic_cell(
    base_struct: Structure,
    min_atoms: int = 80,
    max_atoms: int = 240,
    min_length: float = 10.0,
    force_diagonal: bool = False,
) -> np.ndarray | None:
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
    except BaseException:
        return _ase_cubic(base_struct, min_atoms, max_atoms)
    return cst.transformation_matrix


def _ase_cubic(base_struture, min_atoms: int = 80, max_atoms: int = 240):
    from ase.build import find_optimal_cell_shape, get_deviation_from_optimal_cell_shape
    from pymatgen.io.ase import AseAtomsAdaptor

    _logger.warn("ASE cubic supercell generation.")

    aaa = AseAtomsAdaptor()
    ase_atoms = aaa.get_atoms(base_struture)
    lower = math.ceil(min_atoms / base_struture.num_sites)
    upper = math.floor(max_atoms / base_struture.num_sites)
    min_dev = (float("inf"), None)
    for size in range(lower, upper + 1):
        _logger.warn(f"Trying size {size} out of {upper}.")
        sc = find_optimal_cell_shape(
            ase_atoms.cell, target_size=size, target_shape="sc"
        )
        sc_cell = aaa.get_atoms(base_struture * sc).cell
        deviation = get_deviation_from_optimal_cell_shape(sc_cell, target_shape="sc")
        min_dev = min(min_dev, (deviation, sc))
    if min_dev[1] is None:
        raise RuntimeError("Could not find a cubic supercell")
    return min_dev[1]
