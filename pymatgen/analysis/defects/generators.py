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
from abc import ABCMeta
from itertools import combinations
from typing import Generator

from monty.json import MSONable
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, PeriodicSite, Species, Structure
from pymatgen.io.vasp import Chgcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.analysis.defects.core import Interstitial, Substitution, Vacancy
from pymatgen.analysis.defects.utils import ChargeInsertionAnalyzer, remove_collisions

__author__ = "Jimmy-Xuan Shen"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy-Xuan Shen @jmmshn"
__date__ = "Mar 15, 2022"

logger = logging.getLogger(__name__)


class DefectGenerator(MSONable, metaclass=ABCMeta):
    """Abstract class for a defect generator."""

    def _space_group_analyzer(self, structure: Structure) -> SpacegroupAnalyzer:
        """Get the ``SpaceGroupAnalyzer``."""
        struct = _remove_oxidation_states(structure)
        if hasattr(self, "symprec") and hasattr(self, "angle_tolerance"):
            return SpacegroupAnalyzer(
                struct,
                symprec=self.symprec,
                angle_tolerance=self.angle_tolerance,
            )
        else:
            raise ValueError("This generator does not have symprec and angle_tolerance")

    def _stucture_matcher(self) -> StructureMatcher:
        """Get the ``StructureMatcher``."""
        if (
            hasattr(self, "ltol")
            and hasattr(self, "stol")
            and hasattr(self, "angle_tol")
        ):
            return StructureMatcher(
                ltol=self.ltol, stol=self.stol, angle_tol=self.angle_tol
            )
        else:
            raise ValueError("This generator does not have symprec and angle_tolerance")


class VacancyGenerator(DefectGenerator):
    def __init__(
        self,
        symprec: float = 0.01,
        angle_tolerance: float = 5,
    ):
        """Generator for vacancy defects.

        Args:
            symprec:  Tolerance for symmetry finding.
            (parameter for ``SpacegroupAnalyzer``).
            angle_tolerance: Angle tolerance for symmetry finding.
            (parameter for ``SpacegroupAnalyzer``).
        """
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

    def get_defects(
        self, structure: Structure, rm_species: list[str | Species] = None, **kwargs
    ) -> Generator[Vacancy, None, None]:
        """Generate a vacancy defects.

        Args:
            structure: The bulk structure the vacancies are generated from.
            rm_species: List of species to be removed. If None considered all species.
            **kwargs: Additional keyword arguments for the ``Vacancy`` constructor.

        Returns:
            Generator[Vacancy, None, None]: Generator that yields a list of ``Vacancy`` objects.
        """
        all_species = [*map(str, structure.composition.elements)]

        if rm_species is None:
            rm_species = all_species
        else:
            rm_species = [*map(str, rm_species)]

        if not set(rm_species).issubset(all_species):
            raise ValueError(
                f"rm_species({rm_species}) must be a subset of the structure's species ({all_species})."
            )

        sga = self._space_group_analyzer(structure)
        sym_struct = sga.get_symmetrized_structure()
        for site_group in sym_struct.equivalent_sites:
            site = site_group[0]
            if _element_str(site.specie) in rm_species:
                yield Vacancy(
                    structure=_remove_oxidation_states(structure), site=site, **kwargs
                )


class SubstitutionGenerator(DefectGenerator):
    def __init__(self, symprec: float = 0.01, angle_tolerance: float = 5):
        """Generator of substitutions for symmetry distinct sites in a structure.

        Args:
            symprec:  Tolerance for symmetry finding (parameter for ``SpacegroupAnalyzer``).
            angle_tolerance: Angle tolerance for symmetry finding (parameter for ``SpacegroupAnalyzer``).

        """
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

    def get_defects(
        self, structure: Structure, substitution: dict[str, list[str]], **kwargs
    ) -> Generator[Substitution, None, None]:
        """Generate subsitutional defects.

        Args:
            structure: The bulk structure the vacancies are generated from.
            substitution: The substitutions to be made given as a dictionary.
                e.g. {"Ga": ["Mg", "Ca"]} means that Ga is substituted with Mg or Ca.
            **kwargs: Additional keyword arguments for the ``Substitution`` constructor.

        Returns:
            Generator[Substitution, None, None]: Generator that yields a list of ``Substitution`` objects
        """
        sga = self._space_group_analyzer(structure)
        sym_struct = sga.get_symmetrized_structure()
        for site_group in sym_struct.equivalent_sites:
            site = site_group[0]
            el_str = _element_str(site.specie)
            if el_str not in substitution.keys():
                continue
            for sub_el in substitution[el_str]:
                sub_site = PeriodicSite(
                    Species(sub_el),
                    site.frac_coords,
                    structure.lattice,
                    properties=site.properties,
                )
                yield Substitution(structure, sub_site, **kwargs)


class AntiSiteGenerator(DefectGenerator):
    def __init__(self, symprec: float = 0.01, angle_tolerance: float = 5):
        """Generator of all anti-site defects.

        Args:
            symprec:  Tolerance for symmetry finding (parameter for ``SpacegroupAnalyzer``).
            angle_tolerance: Angle tolerance for symmetry finding (parameter for ``SpacegroupAnalyzer``).

        Returns:
            Generator[Substitution, None, None]: Generator that yields a list of ``Substitution`` objects
        """
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

    def get_defects(
        self,
        structure: Structure,
        **kwargs,
    ) -> Generator[Substitution, None, None]:
        """Generate anti-site defects.

        Args:
            structure: The bulk structure the anti-site defects are generated from.
        """
        all_species = [*map(_element_str, structure.composition.elements)]
        subs = collections.defaultdict(list)
        for u, v in combinations(all_species, 2):
            subs[u].append(v)
            subs[v].append(u)
        logger.debug(f"All anti-site pairings: {subs}")
        return SubstitutionGenerator.get_defects(self, structure, subs)


class InterstitialGenerator(DefectGenerator):
    def __init__(self, min_dist: float = 0.5) -> None:
        """Generator of interstitiald defects.

        Args:
            min_dist: Minimum distance between an interstitial and the nearest atom.
        """
        self.min_dist = min_dist

    def get_defects(
        self, structure: Structure, insertions: dict[str, list[list[float]]], **kwargs
    ) -> Generator[Interstitial, None, None]:
        """Generate interstitials.

        Args:
            structure: The bulk structure the interstitials are generated from.
            insertions: The insertions to be made given as a dictionary {"Mg": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]}.
            **kwargs: Additional keyword arguments for the ``Interstitial`` constructor.

        Returns:
            Generator[Interstitial, None, None]: Generator that yields a list of ``Interstitial`` objects
        """
        for el_str, coords in insertions.items():

            for coord in self._filter_colliding(coords, structure=structure):
                isite = PeriodicSite(
                    species=Species(el_str), coords=coord, lattice=structure.lattice
                )
                yield Interstitial(structure, isite, **kwargs)

    def _filter_colliding(
        self, fcoords: list[list[float]], structure: Structure
    ) -> Generator[list[float], None, None]:
        """Check the sites for collisions.

        Args:
            fcoords: List of fractional coordinates of the sites.
            structure: The bulk structure the interstitials placed in.
        """
        unique_fcoords = set(tuple(f) for f in fcoords)
        cleaned_fcoords = remove_collisions(
            fcoords=list(unique_fcoords), structure=structure, min_dist=self.min_dist
        )
        cleaned_fcoords = set(tuple(f) for f in cleaned_fcoords)
        for fc in fcoords:
            if tuple(fc) not in cleaned_fcoords:
                continue
            yield fc


class ChargeInterstitialGenerator(InterstitialGenerator):
    def __init__(
        self,
        clustering_tol: float = 0.6,
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5,
        min_dist: float = 1.0,
        avg_radius: float = 0.4,
        max_avg_charge: float = 0.9,
    ) -> None:
        """Generator of interstitiald defects.

        Args:
            clustering_tol: Tolerance for clustering see :meth:`pymatgen.analysis.defects.utils.cluster_nodes`.
            ltol: Tolerance for lattice parameter matching
            stol: Tolerance for site matching
            angle_tol: Tolerance for angles in degrees
            min_dist: Minimum to atoms in the host structure
            avg_radius: The radius around each local minima used to evaluate the average charge.
            max_avg_charge: The maximum average charge to accept.
        """
        self.clustering_tol = clustering_tol
        self.ltol = ltol
        self.stol = stol
        self.angle_tol = angle_tol
        self.avg_radius = avg_radius
        self.max_avg_charge = max_avg_charge
        super().__init__(min_dist=min_dist)

    def get_defects(self, chgcar: Chgcar, insert_species: set[str] | list[str], **kwargs) -> Generator[Interstitial, None, None]:  # type: ignore[override]
        """Generate interstitials.

        Args:
            chgcar: The chgcar object to be used for the charge density.
            insert_species: The species to be inserted.
            **kwargs: Additional keyword arguments for the ``Interstitial`` constructor.
        """
        if len(set(insert_species)) != len(insert_species):
            raise ValueError("Insert species must be unique.")
        cand_sites = [*self._get_candidate_sites(chgcar)]
        for species in insert_species:
            yield from super().get_defects(chgcar.structure, {species: cand_sites})

    def _get_candidate_sites(self, chgcar: Chgcar):
        cia = ChargeInsertionAnalyzer(
            chgcar,
            clustering_tol=self.clustering_tol,
            ltol=self.ltol,
            stol=self.stol,
            angle_tol=self.angle_tol,
            min_dist=self.min_dist,
        )
        avg_chg_groups = cia.filter_and_group(
            avg_radius=self.avg_radius, max_avg_charge=self.max_avg_charge
        )
        for _, g in avg_chg_groups:
            yield min(g)


def _element_str(sp_or_el: Species | Element) -> str:
    """Convert a species or element to a string."""
    if isinstance(sp_or_el, Species):
        return str(sp_or_el.element)
    elif isinstance(sp_or_el, Element):
        return str(sp_or_el)
    else:
        raise ValueError(f"{sp_or_el} is not a species or element")


def _remove_oxidation_states(structure: Structure) -> Structure:
    struct = structure.copy()
    struct.remove_oxidation_states()
    return struct
