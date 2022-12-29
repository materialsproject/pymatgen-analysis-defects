"""Extended FNV correction by Kumagai/Oba.

Works as a light wrapper around pydefect.
https://github.com/kumagai-group/pydefect
"""

from __future__ import annotations

import math

from pymatgen.analysis.defects.utils import CorrectionResult

# check that pydefect is installed
try:
    pass
except ImportError:
    raise ImportError("pydefect is not installed. Please install it first.")

import logging

from pydefect.analyzer.calc_results import CalcResults
from pydefect.cli.vasp.make_efnv_correction import make_efnv_correction

# suppress pydefect INFO messages
logging.getLogger("pydefect").setLevel(logging.WARNING)

from pymatgen.core import Structure
from pymatgen.io.vasp import Outcar, Vasprun

__author__ = "Jimmy-Xuan Shen"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy-Xuan Shen"
__email__ = "jmmshn@gmail.com"

_logger = logging.getLogger(__name__)


def read_vasp_output(vasprun: Vasprun, outcar: Outcar) -> CalcResults:
    """Reads vasprun.xml and OUTCAR files and returns a CalcResults object."""
    return CalcResults(
        structure=vasprun.final_structure,
        energy=outcar.final_energy,
        magnetization=outcar.total_mag or 0.0,
        potentials=[-p for p in outcar.electrostatic_potential],
        electronic_conv=vasprun.converged_electronic,
        ionic_conv=vasprun.converged_ionic,
    )
    # TODO: for now electronstatic_potential is not stored in atomate2
    # Once it is we can create a new constructor


def get_kumagai_correction(
    defect_structure: Structure,
    bulk_structure: Structure,
    dielectric_tensor: list[list[float]],
    **kwargs,
) -> CorrectionResult:
    """Returns the Kumagai/Oba EFNV correction for a given defect.

    Args:
        charge: Charge of the defect.
        defect_structure: Defect structure.
        bulk_structure: Bulk structure.
        dielectric_tensor: Dielectric tensor.
        **kwargs: Keyword arguments to pass to `make_efnv_correction`.
    """
    # ensure that the structures have the "potential" site property
    bulk_potentials = [site.properties["potential"] for site in defect_structure]
    defect_potentials = [site.properties["potential"] for site in bulk_structure]

    defect_calc_results = CalcResults(
        structure=defect_structure,
        energy=math.inf,
        magnetization=math.inf,
        potentials=defect_potentials,
    )
    bulk_calc_results = CalcResults(
        structure=bulk_structure,
        energy=math.inf,
        magnetization=math.inf,
        potentials=bulk_potentials,
    )

    efnv_corr = make_efnv_correction(
        calc_results=defect_calc_results,
        bulk_calc_results=bulk_calc_results,
        dielectric_tensor=dielectric_tensor,
        **kwargs,
    )

    return CorrectionResult(
        correction_energy=efnv_corr.correction_energy, metadata={"efnv_corr": efnv_corr}
    )
