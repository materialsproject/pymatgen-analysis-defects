"""Extended FNV correction by Kumagai/Oba.

Works as a light wrapper around pydefect.
https://github.com/kumagai-group/pydefect
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

from pymatgen.analysis.defects.utils import CorrectionResult, get_zfile
from pymatgen.io.vasp import Outcar, Vasprun

if TYPE_CHECKING:
    from pymatgen.core import Structure

try:
    from vise import user_settings

    # Disable messages from pydefect import
    user_settings.logger.setLevel(logging.CRITICAL)
    from pydefect.analyzer.calc_results import CalcResults
    from pydefect.cli.vasp.make_efnv_correction import make_efnv_correction

    __has_pydefect__ = True
except ImportError:  # pragma: no cover
    __has_pydefect__ = False

__author__ = "Jimmy-Xuan Shen"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy-Xuan Shen"
__email__ = "jmmshn@gmail.com"
_logger = logging.getLogger(__name__)
# suppress pydefect INFO messages
logging.getLogger("pydefect").setLevel(logging.WARNING)


def _check_import_pydefect() -> None:
    """Import pydefect if it is installed."""
    if not __has_pydefect__:
        msg = "vise/pydefect is not installed. Please install it first."
        raise ModuleNotFoundError(
            msg,
        )


def get_structure_with_pot(directory: Path) -> Structure:
    """Reads vasprun.xml and OUTCAR files in a directory.

    Args:
        directory: Directory containing vasprun.xml and OUTCAR files.

    Returns:
        Structure with "potential" site property.
    """
    _check_import_pydefect()
    d_ = Path(directory)
    f_vasprun = get_zfile(d_, "vasprun.xml")
    f_outcar = get_zfile(d_, "OUTCAR")
    vasprun = Vasprun(f_vasprun)
    outcar = Outcar(f_outcar)

    calc = CalcResults(
        structure=vasprun.final_structure,
        energy=outcar.final_energy,
        magnetization=outcar.total_mag or 0.0,
        potentials=[-p for p in outcar.electrostatic_potential],
        electronic_conv=vasprun.converged_electronic,
        ionic_conv=vasprun.converged_ionic,
    )

    return calc.structure.copy(site_properties={"potential": calc.potentials})


def get_efnv_correction(
    charge: int,
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
    _check_import_pydefect()
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
        charge=charge,
        calc_results=defect_calc_results,
        perfect_calc_results=bulk_calc_results,
        dielectric_tensor=dielectric_tensor,
        **kwargs,
    )

    return CorrectionResult(
        correction_energy=efnv_corr.correction_energy,
        metadata={"efnv_corr": efnv_corr},
    )
