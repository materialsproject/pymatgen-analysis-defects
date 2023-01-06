from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from monty.json import MSONable


class CorrectionType(Enum):
    """Correction type"""

    electrostatic = "electrostatic"
    potential_alignment = "potential_alignment"


@dataclass
class Correction(MSONable):
    """Summary of a single energy correction

    Attributes:
        correction_energy: float
        correction_type: CorrectionType
        metadata: any metadata (generally used to store pot align plot data)
    """

    correction_energy: float
    correction_type: CorrectionType
    uncertainty: float = (None,)
    name: str = ("Defect correction",)
    description: str = ("",)
    metadata: Optional[dict[Any, Any]] = None

    def __post_init__(self):
        self.metadata: dict = {} if self.metadata is None else self.metadata


@dataclass
class CorrectionsSummary(MSONable):
    """A summary of all defect corrections applied to a structure.

    Attributes:
        corrections: A dictionary mapping CorrectionType to its correction energy
        metadata: A dictionary of metadata for plotting and intermediate analysis.
    """

    corrections: dict[CorrectionType, float]
    metadata: dict[Any, Any]

    @property
    def correction_energy(self):
        """Get total correction energy"""
        return sum(self.corrections.values())

    @classmethod
    def from_corrections(cls, corrections: list[Correction]):
        """Join many corrections. If many corrections of
        the same type are present, only the last will be
        included.

        Parameters
        -------------

        corrections
            List of Corrections.
        """
        corrs = {}
        metadata = {}
        for c in corrections:
            corrs.update({c.correction_type: c.correction_energy})
            metadata.update(c.metadata)
        return CorrectionsSummary(corrs, metadata)
