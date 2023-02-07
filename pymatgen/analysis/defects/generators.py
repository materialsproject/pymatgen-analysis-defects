"""Defect generators."""

from __future__ import annotations

import collections
import logging
from abc import ABCMeta
from itertools import combinations
from typing import Generator

from monty.json import MSONable
from pymatgen.core import Element, PeriodicSite, Species, Structure
from pymatgen.io.vasp import Chgcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.analysis.defects.core import Defect, Interstitial, Substitution, Vacancy
from pymatgen.analysis.defects.utils import (
    ChargeInsertionAnalyzer,
    TopographyAnalyzer,
    remove_collisions,
)

__author__ = "Jimmy-Xuan Shen"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy-Xuan Shen @jmmshn"
__date__ = "Mar 15, 2022"

_logger = logging.getLogger(__name__)


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

    def get_defects(self, *args, **kwargs) -> list[Defect]:
        """Call the generator and convert the results into a list."""
        return list(self.generate(*args, **kwargs))


class VacancyGenerator(DefectGenerator):
    """Generator for vacancy defects.

    Attributes:
        symprec:  Tolerance for symmetry finding
            (parameter for ``SpacegroupAnalyzer``).
        angle_tolerance: Angle tolerance for symmetry finding
            (parameter for ``SpacegroupAnalyzer``).
    """

    def __init__(
        self,
        symprec: float = 0.01,
        angle_tolerance: float = 5,
    ):
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

    def generate(
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
    """Generator of substitutions for symmetry distinct sites in a structure.

    Attributes:
        symprec:  Tolerance for symmetry finding (parameter for ``SpacegroupAnalyzer``).
        angle_tolerance: Angle tolerance for symmetry finding (parameter for ``SpacegroupAnalyzer``).

    """

    def __init__(self, symprec: float = 0.01, angle_tolerance: float = 5):
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

    def generate(
        self, structure: Structure, substitution: dict[str, str | list], **kwargs
    ) -> Generator[Substitution, None, None]:
        """Generate subsitutional defects.

        Args:
            structure: The bulk structure the vacancies are generated from.
            substitution: The substitutions to be made given as a dictionary.
                e.g. {"Ga": "Ca"} means that Ga is substituted with Ca. You
                can also specify a list of elements to substitute with.
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
            sub_el = substitution[el_str]
            if isinstance(sub_el, str):
                sub_site = PeriodicSite(
                    Species(sub_el),
                    site.frac_coords,
                    structure.lattice,
                    properties=site.properties,
                )
                yield Substitution(structure, sub_site, **kwargs)
            elif isinstance(sub_el, list):
                for el in sub_el:
                    sub_site = PeriodicSite(
                        Species(el),
                        site.frac_coords,
                        structure.lattice,
                        properties=site.properties,
                    )
                    yield Substitution(structure, sub_site, **kwargs)


class AntiSiteGenerator(DefectGenerator):
    """Generator of all anti-site defects.

    Attributes:
        symprec:  Tolerance for symmetry finding (parameter for ``SpacegroupAnalyzer``).
        angle_tolerance: Angle tolerance for symmetry finding (parameter for ``SpacegroupAnalyzer``).
    """

    def __init__(self, symprec: float = 0.01, angle_tolerance: float = 5):
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self._sub_gen = SubstitutionGenerator(symprec, angle_tolerance)

    def generate(
        self,
        structure: Structure,
        **kwargs,
    ) -> Generator[Substitution, None, None]:
        """Generate anti-site defects.

        Args:
            structure: The bulk structure the anti-site defects are generated from.
            **kwargs: Additional keyword arguments for the ``Substitution.generate`` function.
        """
        all_species = [*map(_element_str, structure.composition.elements)]
        subs = collections.defaultdict(list)
        for u, v in combinations(all_species, 2):
            subs[u].append(v)
            subs[v].append(u)
        _logger.debug(f"All anti-site pairings: {subs}")
        for site, species in subs.items():
            for sub in species:
                yield from self._sub_gen.generate(structure, {site: sub}, **kwargs)


class InterstitialGenerator(DefectGenerator):
    """Generator of interstitiald defects.

    Attributes:
        min_dist: Minimum distance between an interstitial and the nearest atom.
    """

    def __init__(self, min_dist: float = 0.5) -> None:
        self.min_dist = min_dist

    def generate(
        self,
        structure: Structure,
        insertions: dict[str, list[list[float]]],
        multiplicies: dict[str, list[int]] | None = None,
        **kwargs,
    ) -> Generator[Interstitial, None, None]:
        """Generate interstitials.

        Args:
            structure: The bulk structure the interstitials are generated from.
            insertions: The insertions to be made given as a dictionary {"Mg": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]}.
            multiplicies: The multiplicities of the insertions to be made given as a dictionary {"Mg": [1, 2]}.
            **kwargs: Additional keyword arguments for the ``Interstitial`` constructor.

        Returns:
            Generator[Interstitial, None, None]: Generator that yields a list of ``Interstitial`` objects
        """
        if multiplicies is None:
            multiplicies = {
                el_str: [1] * len(coords) for el_str, coords in insertions.items()
            }

        for el_str, coords in insertions.items():
            for i, coord in enumerate(
                self._filter_colliding(coords, structure=structure)
            ):
                mul = multiplicies[el_str][i]
                isite = PeriodicSite(
                    species=Species(el_str), coords=coord, lattice=structure.lattice
                )
                yield Interstitial(structure, isite, multiplicity=mul, **kwargs)

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


class VoronoiInterstitialGenerator(InterstitialGenerator):
    """Generator for interstitials based on a simple Voronoi.

    Attributes:
        clustering_tol: Tolerance for clustering the Voronoi nodes.
        min_dist: Minimum distance between an interstitial and the nearest atom.
        ltol: Tolerance for lattice matching.
        stol: Tolerance for structure matching.
        angle_tol: Angle tolerance for structure matching.
        kwargs: Additional keyword arguments for the ``TopographyAnalyzer`` constructor.
    """

    def __init__(
        self,
        clustering_tol: float = 0.5,
        min_dist: float = 0.9,
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5,
        **kwargs,
    ) -> None:
        self.clustering_tol = clustering_tol
        self.min_dist = min_dist
        self.ltol = ltol
        self.stol = stol
        self.angle_tol = angle_tol
        self.top_kwargs = kwargs
        super().__init__()

    def generate(self, structure: Structure, insert_species: set[str] | list[str], **kwargs) -> Generator[Interstitial, None, None]:  # type: ignore[override]
        """Generate interstitials.

        Args:
            structure: The bulk structure the interstitials inserted in.
            insert_species: The species to be inserted.
            **kwargs: Additional keyword arguments for the ``Interstitial`` constructor.
        """
        if len(set(insert_species)) != len(insert_species):
            raise ValueError("Insert species must be unique.")
        cand_sites_and_mul = [*self._get_candidate_sites(structure)]
        for species in insert_species:
            cand_sites = [cand_site for cand_site, mul in cand_sites_and_mul]
            multiplicity = [mul for cand_site, mul in cand_sites_and_mul]
            yield from super().generate(
                structure,
                insertions={species: cand_sites},
                multiplicies={species: multiplicity},
                **kwargs,
            )

    def _get_candidate_sites(
        self, structure: Structure
    ) -> Generator[tuple[list[float], int], None, None]:
        """Get the candidate sites for interstitials.

        Args:
            structure: The bulk structure the interstitials inserted in.
        """
        framework = list(structure.symbol_set)
        top = TopographyAnalyzer(
            structure, framework, [], check_volume=False, **self.top_kwargs
        )
        insert_sites = dict()
        multiplicity: dict[int, int] = dict()
        for fpos, lab in top.labeled_sites:
            if lab in insert_sites:
                multiplicity[lab] += 1
                continue
            insert_sites[lab] = fpos
            multiplicity[lab] = 1

        for key in insert_sites.keys():
            yield insert_sites[key], multiplicity[key]


class ChargeInterstitialGenerator(InterstitialGenerator):
    """Generator of interstitiald defects.

    Attributes:
        clustering_tol: Tolerance for clustering see :meth:`pymatgen.analysis.defects.utils.cluster_nodes`.
        ltol: Tolerance for lattice parameter matching
        stol: Tolerance for site matching
        angle_tol: Tolerance for angles in degrees
        min_dist: Minimum to atoms in the host structure
        avg_radius: The radius around each local minima used to evaluate the average charge.
        max_avg_charge: The maximum average charge to accept.
    """

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
        self.clustering_tol = clustering_tol
        self.ltol = ltol
        self.stol = stol
        self.angle_tol = angle_tol
        self.avg_radius = avg_radius
        self.max_avg_charge = max_avg_charge
        super().__init__(min_dist=min_dist)

    def generate(self, chgcar: Chgcar, insert_species: set[str] | list[str], **kwargs) -> Generator[Interstitial, None, None]:  # type: ignore[override]
        """Generate interstitials.

        Args:
            chgcar: The chgcar object to be used for the charge density.
            insert_species: The species to be inserted.
            **kwargs: Additional keyword arguments for the ``Interstitial`` constructor.
        """
        if len(set(insert_species)) != len(insert_species):
            raise ValueError("Insert species must be unique.")
        cand_sites_and_mul = [*self._get_candidate_sites(chgcar)]
        for species in insert_species:
            cand_sites = [cand_site for cand_site, mul in cand_sites_and_mul]
            multiplicity = [mul for cand_site, mul in cand_sites_and_mul]
            yield from super().generate(
                chgcar.structure,
                insertions={species: cand_sites},
                multiplicies={species: multiplicity},
                **kwargs,
            )

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
            yield min(g), len(g)


def _element_str(sp_or_el: Species | Element) -> str:
    """Convert a species or element to a string."""
    if isinstance(sp_or_el, Species):
        return str(sp_or_el.element)
    elif isinstance(sp_or_el, Element):
        return str(sp_or_el)
    else:
        raise ValueError(f"{sp_or_el} is not a species or element")  # pragma: no cover


def _remove_oxidation_states(structure: Structure) -> Structure:
    """Get a structure with oxidation states removed."""
    struct = structure.copy()
    struct.remove_oxidation_states()
    return struct
