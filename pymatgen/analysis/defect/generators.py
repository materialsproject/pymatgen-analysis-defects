"""Defect generators (bulk structure and other inputs) -> (Defect Objects)."""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Iterable

from monty.json import MSONable
from pymatgen.core import PeriodicSite, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.analysis.defect.core import Defect, Vacancy

__author__ = "Jimmy-Xuan Shen"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy-Xuan Shen @jmmshn"
__date__ = "Mar 15, 2022"


class DefectGenerator(MSONable, metaclass=ABCMeta):
    """Abstract class for a defect generator."""

    def __init__(self, structure: Structure):
        """Initialize a defect generator.

        Args:
            structure: The bulk structure the defects are generated from.
        """
        self.structure = structure
        # make sure we only look at structure symmetries
        self.structure.remove_oxidation_states()

    def __iter__(self):
        """Iterate over the defect objects."""
        for defect in self.generate_defects():
            yield defect

    @abstractmethod
    def get_defect(self, site: PeriodicSite) -> Defect:
        """Return a defect object for a given site object."""

    @abstractmethod
    def generate_defects(self) -> Iterable[Defect]:
        """Generate a list of symmetry-distinct site objects, the yield their corresponding defect objects."""


class VacancyGenerator(DefectGenerator):
    """Generate vacancy for each symmtry distinct site in a structure."""

    def __init__(self, structure: Structure, rm_species: list[str] | None = None):
        """Initialize a vacancy generator.

        Args:
            structure: The bulk structure the vacancies are generated from.
            rm_species: The species to be removed.
        """
        super().__init__(structure)
        all_species = [*map(str, structure.composition.elements)]
        self.rm_species = rm_species
        self.rm_species = [*map(str, rm_species)] if rm_species is not None else all_species

    def get_defect(self, site: PeriodicSite) -> Vacancy:
        """Generate a vacancy object.

        Args:
            site: The site of the vacancy.

        Returns:
            Vacancy: The vacancy object.
        """
        return Vacancy(self.structure, site)

    def generate_defects(self) -> Iterable[Defect]:
        """Generate the vacancy objects."""
        sga = SpacegroupAnalyzer(self.structure)
        sym_struct = sga.get_symmetrized_structure()
        for site_group in sym_struct.equivalent_sites:
            site = site_group[0]
            if site.specie.symbol in self.rm_species:
                yield self.get_defect(site)
