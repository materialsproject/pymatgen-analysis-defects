"""Defect generators (bulk structure and other inputs) -> (Defect Objects)."""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Iterable

from monty.json import MSONable
from pymatgen.core import Element, PeriodicSite, Species, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.analysis.defects.core import Defect, Substitution, Vacancy

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
    def generate_defects(self) -> Iterable[Defect]:
        """
        Generate a list of symmetry-distinct site objects, the yield their corresponding
        defect objects.
        """


class VacancyGenerator(DefectGenerator):
    """Generate vacancy for each symmetry distinct site in a structure."""

    def __init__(self, structure: Structure, rm_species: list[str] | None = None):
        """Initialize a vacancy generator.

        Args:
            structure: The bulk structure the vacancies are generated from.
            rm_species: The species to be removed.
        """
        super().__init__(structure)
        all_species = [*map(str, structure.composition.elements)]
        self.rm_species = (
            [*map(str, rm_species)] if rm_species is not None else all_species
        )

    def generate_defects(self) -> Iterable[Defect]:
        """Generate the vacancy objects."""
        sga = SpacegroupAnalyzer(self.structure)
        sym_struct = sga.get_symmetrized_structure()
        for site_group in sym_struct.equivalent_sites:
            site = site_group[0]
            if element_str(site.specie) in self.rm_species:
                yield Vacancy(self.structure, site)


class SubstitutionGenerator(DefectGenerator):
    """Generate substitution for symmetry distinct sites in a structure."""

    def __init__(self, structure: Structure, substitution: dict[str, str]):
        """Initialize a substitution generator.

        Args:
            structure: The bulk structure the vacancies are generated from.
            substitution: The substitutions to be made given as a dictionary.
                e.g. {"Mg": "Ga"} means that Mg is substituted on the Ga site.
        """
        super().__init__(structure)
        self.substitution = substitution

    def generate_defects(self) -> Iterable[Defect]:
        """Generate the substitution objects."""
        sga = SpacegroupAnalyzer(self.structure)
        sym_struct = sga.get_symmetrized_structure()
        for site_group in sym_struct.equivalent_sites:
            site = site_group[0]
            el_str = element_str(site.specie)
            if el_str not in self.substitution.keys():
                continue
            new_element = self.substitution[el_str]
            sub_site = PeriodicSite(
                Species(new_element),
                site.frac_coords,
                self.structure.lattice,
                properties=site.properties,
            )
            yield Substitution(self.structure, sub_site)


def element_str(sp_or_el: Species | Element) -> str:
    """Convert a species or element to a string."""
    if isinstance(sp_or_el, Species):
        return str(sp_or_el.element)
    elif isinstance(sp_or_el, Element):
        return str(sp_or_el)
    else:
        raise ValueError(f"{sp_or_el} is not a species or element")
