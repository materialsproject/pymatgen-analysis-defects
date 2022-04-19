"""Base classes representing defects."""
from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod

import numpy as np
from monty.json import MSONable
from numpy.typing import ArrayLike
from pymatgen.core.structure import PeriodicSite, Structure

__author__ = "Jimmy-Xuan Shen"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy Shen @jmmshn"
__status__ = "Development"
__date__ = "Mar 15, 2022"

logger = logging.getLogger(__name__)


class Defect(MSONable, metaclass=ABCMeta):
    """Abstract class for a single point defect."""

    def __init__(
        self, structure: Structure, pos: ArrayLike[np.float_], charge: int = 0, multiplicity: int | None = None
    ) -> None:
        """Initialize a defect object.

        Args:
            structure: The structure of the defect.
            pos: The position of the defect.
            charge: The charge of the defect.
            multiplicity: The multiplicity of the defect.

        """
        self.structure = structure
        self.pos = pos
        self.charge = charge
        self.multiplicity = multiplicity if multiplicity is not None else self.get_multiplicity()

    @abstractmethod
    def get_multiplicity(self) -> int:
        """Get the multiplicity of the defect.

        Returns:
            int: The multiplicity of the defect.
        """


class Vacancy(Defect):
    """Class representing a vacancy defect."""

    def __init__(
        self, structure: Structure, site: PeriodicSite, charge: int = 0, multiplicity: int | None = None
    ) -> None:
        """Initialize a vacancy defect object."""
        super().__init__(structure, site, charge, multiplicity)

    def get_multiplicity(self) -> int:
        """Get the multiplicity of the vacancy."""
