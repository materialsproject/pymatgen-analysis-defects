"""Defect generators.

The generator objects can be used directly as generators once instantiated.

.. code-block:: python

    from pymatgen.analysis.defects.generators import VacancyGenerator

    gen = VacancyGenerator(structure, ["Ga", "N"])
    for defect in gen:
        # do something with defect
        pass

In this case the generator's ``generate_defects`` method is called with only the default values.

For more fine-grained control, the generator's ``generate_defects`` method can be called directly.
With non-default values.

.. code-block:: python

    from pymatgen.analysis.defects.generators import VacancyGenerator

    gen = VacancyGenerator(structure, ["Ga", "N"])
    for defect in gen.generate_defects(max_avg_charge=0.5):
        # do something with defect
        pass

"""

from __future__ import annotations

import collections
import logging
from abc import ABCMeta, abstractmethod
from itertools import combinations
from typing import Generator

from monty.json import MSONable
from pymatgen.core import Element, PeriodicSite, Species, Structure
from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.analysis.defects.core import Defect, Interstitial, Substitution, Vacancy
from pymatgen.analysis.defects.utils import ChargeInsertionAnalyzer

__author__ = "Jimmy-Xuan Shen"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy-Xuan Shen @jmmshn"
__date__ = "Mar 15, 2022"

logger = logging.getLogger(__name__)


class DefectGenerator(MSONable, metaclass=ABCMeta):
    def __init__(self, structure: Structure, **kwargs):
        """Abstract class for a defect generator.

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
    def generate_defects(self, **kwargs) -> Generator[Defect, None, None]:
        """Generate a list of symmetry-distinct site objects,

        Args:
            **kwargs: Additional keyword arguments for the ``Defect`` constructor.

        Returns:
            Generator[Defect, None, None]: Generator that yields a list of ``Defect`` objects
        """


class VacancyGenerator(DefectGenerator):
    def __init__(self, structure: Structure, rm_species: list[str | Species] = None):
        """Generate vacancy for each symmetry distinct site in a structure.

        Args:
            structure: The bulk structure the vacancies are generated from.
            rm_species: List of species to be removed. If None considered all species.
        """
        super().__init__(structure)
        all_species = [*map(str, structure.composition.elements)]
        if rm_species is None:
            self.rm_species = all_species
        else:
            self.rm_species = [*map(str, rm_species)]

        if not set(self.rm_species).issubset(all_species):
            raise ValueError(
                f"rm_species({rm_species}) must be a subset of the structure's species ({all_species})."
            )

    def generate_defects(
        self, symprec: float = 0.01, angle_tolerance: float = 5, **kwargs
    ) -> Generator[Vacancy, None, None]:
        """Generate a vacancy defects.

        Args:
            symprec:  Tolerance for symmetry finding.
            (parameter for ``SpacegroupAnalyzer``).
            angle_tolerance: Angle tolerance for symmetry finding.
            (parameter for ``SpacegroupAnalyzer``).
            **kwargs: Additional keyword arguments for the ``Defect`` constructor.

        Returns:
            Generator[Vacancy, None, None]: Generator that yields a list of ``Defect`` objects
        """
        sga = SpacegroupAnalyzer(
            self.structure, symprec=symprec, angle_tolerance=angle_tolerance
        )
        sym_struct = sga.get_symmetrized_structure()
        for site_group in sym_struct.equivalent_sites:
            site = site_group[0]
            if _element_str(site.specie) in self.rm_species:
                yield Vacancy(self.structure, site, **kwargs)


class SubstitutionGenerator(DefectGenerator):
    def __init__(self, structure: Structure, substitution: dict[str, list[str]]):
        """Generator of substitutions for symmetry distinct sites in a structure.

        Args:
            structure: The bulk structure the vacancies are generated from.
            substitution: The substitutions to be made given as a dictionary.
                e.g. {"Ga": ["Mg", "Ca"]} means that Ga is substituted with Mg or Ca.
        """
        super().__init__(structure)
        self.substitution = substitution

    def generate_defects(
        self, symprec: float = 0.01, angle_tolerance: float = 5, **kwargs
    ) -> Generator[Substitution, None, None]:
        """Generate a subsitutional defects.

        Args:
            symprec:  Tolerance for symmetry finding (parameter for ``SpacegroupAnalyzer``).
            angle_tolerance: Angle tolerance for symmetry finding (parameter for ``SpacegroupAnalyzer``).
            **kwargs: Additional keyword arguments for the ``Defect`` constructor.

        Returns:
            Generator[Substitution, None, None]: Generator that yields a list of ``Substitution`` objects
        """
        sga = SpacegroupAnalyzer(
            self.structure, symprec=symprec, angle_tolerance=angle_tolerance
        )
        sym_struct = sga.get_symmetrized_structure()
        for site_group in sym_struct.equivalent_sites:
            site = site_group[0]
            el_str = _element_str(site.specie)
            if el_str not in self.substitution.keys():
                continue
            for sub_el in self.substitution[el_str]:
                sub_site = PeriodicSite(
                    Species(sub_el),
                    site.frac_coords,
                    self.structure.lattice,
                    properties=site.properties,
                )
                yield Substitution(self.structure, sub_site, **kwargs)


class AntiSiteGenerator(SubstitutionGenerator):
    """Generate all anti-site defects."""

    def __init__(self, structure: Structure):
        """Initialize an anti-site generator.

        Args:
            structure: The bulk structure the anti-site defects are generated from.
        """
        all_species = [*map(_element_str, structure.composition.elements)]
        subs = collections.defaultdict(list)
        for u, v in combinations(all_species, 2):
            subs[u].append(v)
            subs[v].append(u)
        logger.debug(f"All anti-site pairings: {subs}")
        super().__init__(structure, subs)


class InterstitialGenerator(DefectGenerator):
    def __init__(
        self,
        structure: Structure,
        insertions: dict[str | Species | Element, list[list[float]]],
    ):
        """Generator for intersitials defects in structure.

        Args:
            structure: The bulk structure the interstitials atoms are placed in.
            insertions: Insertions in the form of a dictionary.
                e.g. ``{"Mg": [[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]}`` means that we want to generate
                two interstitials at [0.5, 0.5, 0.5] and [0.25, 0.25, 0.25].

        """
        super().__init__(structure)
        self.insertions = insertions

    @classmethod
    def from_chgcar(
        cls,
        chgcar: Chgcar,
        i_species: str | Species | Element,
        avg_radius: float = 0.4,
        max_avg_charge: float = 1.0,
        n_groups: int = None,
        **kwargs,
    ) -> InterstitialGenerator:
        """Generate interstitials from a CHGCAR object.

        Identify the interstitial sites from the CHGCAR object using ``ChargeInsertionAnalyzer``.
        ``ChargeInsertionAnalyzer`` examines the local minmina in the charge density and organizes
        them into symmetry equivalent groups.
        The groups are ranked by the average charge density in ``avg_radius`` around each site,
        we will filter out groups with average charge density higher than ``max_avg_charge``.
        If ``n_groups`` is specified, we will only select the ``n_groups`` groups with the lowest
        average charge density.

        Args:
            chgcar: The CHGCAR object.
            insertions: Insertions in the form of a dictionary.
                e.g. ``{"Mg": [[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]}`` means that we want to generate
                two interstitials at [0.5, 0.5, 0.5] and [0.25, 0.25, 0.25].
            avg_radius: The radius used to calculate average charge density.
            max_avg_charge: Do no consider local minmas with avg charge above this value.
            n_groups: Only consider the n_groups with lowest average charge density. If None, all groups are considered.
            **kwargs: Other keyword arguments passed to ``ChargeInsertionAnalyzer``.

        Returns:
            An ``InterstitialGenerator`` object.
        """
        cia = ChargeInsertionAnalyzer(chgcar, **kwargs)
        insert_groups = cia.filter_and_group(
            avg_radius=avg_radius, max_avg_charge=max_avg_charge
        )
        i_pos = [group[0] for _, group in insert_groups]
        if n_groups is not None:
            i_pos = i_pos[:n_groups]
        return cls(cia.chgcar.structure.copy(), insertions={i_species: i_pos})

    def generate_defects(self, **kwargs) -> Generator[Interstitial, None, None]:
        """Generate interstitials defects.

        Args:
            **kwargs: Additional keyword arguments for the ``Interstitial`` constructor.

        Returns:
            Generator[Interstitial, None, None]: Generator that yields a list of ``Interstitial`` objects
        """
        for el_str, pos_list in self.insertions.items():
            for pos in pos_list:
                isite = PeriodicSite(Species(el_str), pos, self.structure.lattice)
                yield Interstitial(self.structure, isite, **kwargs)


def _element_str(sp_or_el: Species | Element) -> str:
    """Convert a species or element to a string."""
    if isinstance(sp_or_el, Species):
        return str(sp_or_el.element)
    elif isinstance(sp_or_el, Element):
        return str(sp_or_el)
    else:
        raise ValueError(f"{sp_or_el} is not a species or element")
