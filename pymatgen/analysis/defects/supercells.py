"""Functions for creating supercells for defect calculations."""

import logging
from typing import List, Optional

# from ase.build import find_optimal_cell_shape, get_deviation_from_optimal_cell_shape
# from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

__author__ = "Jimmy Shen"
__copyright__ = "Copyright 2019, The Materials Project"
__maintainer__ = "Jimmy Shen"
__email__ = "jmmshn@gmail.com"
__date__ = "Feb 11, 2021"

logger = logging.getLogger(__name__)

# Helper functions for MigraionHop.get_sc_struture


def get_sc_fromstruct(
    base_struct: Structure,
    min_atoms: int = 80,
    max_atoms: int = 240,
    min_length: float = 10.0,
) -> List[List[int]]:
    """Generate the best supercell from a unitcell.

    The CubicSupercellTransformation from PMG is much faster but don't iterate over as many
    supercell configurations so it's less able to find the best configuration in a give cell size.
    We try the PMG's cubic supercell transformation with a cap on the number of atoms (max_atoms).
    The min_length is decreased by 10% (geometrically) until a supercell can be constructed.

    Args:
        base_struct: structure of the unit cell
        max_atoms: Maximum number of atoms allowed in the supercell.
        min_atoms: Minimum number of atoms allowed in the supercell.
        min_length: Minimum length of the smallest supercell lattice vector.

    Returns:
        struc_sc: Supercell that is as close to cubic as possible
    """
    m_len = min_length
    struct_sc = None
    while struct_sc is None:
        struct_sc = _get_sc_from_struct_pmg(base_struct, min_atoms, max_atoms, m_len)
        max_atoms += 1
    return struct_sc


def _get_sc_from_struct_pmg(
    base_struct: Structure,
    min_atoms: int = 80,
    max_atoms: int = 240,
    min_length: float = 10.0,
) -> Optional[List[List[int]]]:
    """Generate the best supercell from a unitcell using the pymatgen CubicSupercellTransformation.

    Args:
        base_struct: structure of the unit cell
        max_atoms: Maximum number of atoms allowed in the supercell.
        min_atoms: Minimum number of atoms allowed in the supercell.
        min_length: Minimum length of the smallest supercell lattice vector.

    Returns:
        3x3 matrix: supercell matrix
    """
    cst = CubicSupercellTransformation(min_atoms=min_atoms, max_atoms=max_atoms, min_length=min_length)

    try:
        cst.apply_transformation(base_struct)
    except BaseException:
        return None
    return cst.transformation_matrix
