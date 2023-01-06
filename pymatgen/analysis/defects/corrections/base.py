"""Objects for representing corrections to defect calculations."""

from typing import Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from monty.json import MSONable


class CorrectionType(Enum):
    """Correction type."""

    ELECTROSTATIC = "electrostatic"
    POTENTIAL_ALIGNMENT = "potential_alignment"



@dataclass
class Correction(MSONable):
    """Summary of a single energy correction.

    Parameters
    -------------
        correction_energy
            energy (in eV) of the correction
        correction_type
            Enum for the type of correction
        metadata
            Any metadata for this correction
    """

    correction_energy: float
    correction_type: CorrectionType
    uncertainty: float = np.nan
    name: str = "Defect correction"
    description: str = ""
    metadata: Optional[dict[Any, Any]] = None

    def __post_init__(self):
        """Post initialization modifications."""
        self.metadata: dict = {} if self.metadata is None else self.metadata


@dataclass
class CorrectionsSummary(MSONable):
    """A summary of all defect corrections applied to a structure.

    Parameters
    -------------
        corrections
            A dictionary mapping CorrectionType to its correction energy
        metadata
            A dictionary of metadata for plotting and intermediate analysis.
    """

    corrections: dict[CorrectionType, float]
    metadata: dict[Any, Any]

    @property
    def correction_energy(self):
        """Get total correction energy."""
        return sum(self.corrections.values())

    @classmethod
    def from_corrections(cls, corrections: list[Correction]):
        """Join many corrections.

        Args:
            corrections: List of Corrections.
        """
        corrs = {}
        metadata = {}
        for corr in corrections:
            corrs.update({corr.correction_type: corr.correction_energy})
            metadata.update(corr.metadata)
        return CorrectionsSummary(corrs, metadata)
