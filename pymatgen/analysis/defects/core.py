"""Classes representing defects."""
from __future__ import annotations

import collections
import logging
from abc import ABCMeta, abstractmethod, abstractproperty
from enum import Enum
from typing import Dict

import numpy as np
from monty.json import MSONable
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import Element, PeriodicSite, Species, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure

from pymatgen.analysis.defects.supercells import get_sc_fromstruct

# TODO Possible redesign idea: ``DefectSite`` class defined with a defect object.
# This makes some of the accounting logic a bit harder since we will probably
# just have one concrete ``Defect`` class so you can write custom multiplicity functions
# but it makes the implementation of defect complexes trivial.
# i.e. each defect will be defined by a structure and a collection of ``DefectSite``s

__author__ = "Jimmy-Xuan Shen"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy-Xuan Shen @jmmshn"
__date__ = "Mar 15, 2022"

_logger = logging.getLogger(__name__)


class DefectType(Enum):
    """Defect type, for sorting purposes."""

    Vacancy = 0
    Substitution = 1
    Interstitial = 2
    Other = 3


class Defect(MSONable, metaclass=ABCMeta):
    """Abstract class for a single point defect."""

    def __init__(
        self,
        structure: Structure,
        site: PeriodicSite,
        multiplicity: int | None = None,
        oxi_state: float | None = None,
        symprec: float = 0.01,
        angle_tolerance: float = 5,
        user_charges: list[int] | None = None,
    ) -> None:
        """Initialize a defect object.

        Args:
            structure: The structure of the defect.
            site: The site
            multiplicity: The multiplicity of the defect.
            oxi_state: The oxidation state of the defect, if not specified,
                this will be determined automatically.
            symprec: Tolerance for symmetry finding.
            angle_tolerance: Angle tolerance for symmetry finding.
            user_charges: User specified charge states. If specified,
                ``get_charge_states`` will return this list. If ``None`` or empty list
                the charge states will be determined automatically.
        """
        self.structure = structure
        self.site = site
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self.multiplicity = (
            multiplicity if multiplicity is not None else self.get_multiplicity()
        )
        self.user_charges = user_charges if user_charges else []
        if oxi_state is None:
            # TODO this step might take time so wrap it in a timer
            self.structure.add_oxidation_state_by_guess()
            self.oxi_state = self._guess_oxi_state()
        else:
            self.oxi_state = oxi_state

    @abstractmethod
    def get_multiplicity(self) -> int:
        """Get the multiplicity of the defect.

        Returns:
            int: The multiplicity of the defect.
        """

    @abstractmethod
    def _guess_oxi_state(self) -> float:
        """Best guess for the oxidation state of the defect.

        Returns:
            float: The oxidation state of the defect.
        """

    @abstractmethod
    def __repr__(self) -> str:
        """Representation of the defect."""

    @abstractproperty
    def name(self) -> str:
        """Name of the defect."""

    @abstractproperty
    def defect_structure(self) -> Structure:
        """Get the unit-cell structure representing the defect."""

    @abstractproperty
    def element_changes(self) -> Dict[Element, int]:
        """Get the species changes of the defect.

        Returns:
            Dict[Element, int]: The species changes of the defect.
        """

    def get_charge_states(self, padding: int = 1) -> list[int]:
        """Potential charge states for a given oxidation state.

        If user charges are specified, these will be returned.
        Otherwise, the charge states will be determined automatically based
        on the oxidation state with a padding on either sites of 0 and the
        oxidation state value.

        Args:
            padding: The number of charge states on the on either side of
                0 and the oxidation state.

        Returns:
            list of possible charge states
        """
        if self.user_charges:
            return self.user_charges

        if isinstance(self.oxi_state, int) or self.oxi_state.is_integer():
            oxi_state = int(self.oxi_state)
        else:
            raise ValueError("Oxidation state must be an integer")

        if oxi_state >= 0:
            charges = [*range(-padding, oxi_state + padding + 1)]
        else:
            charges = [*range(oxi_state - padding, padding + 1)]

        return charges

    def get_supercell_structure(
        self,
        sc_mat: np.ndarray | None = None,
        dummy_species: str | None = None,
        min_atoms: int = 80,
        max_atoms: int = 240,
        min_length: float = 10.0,
        force_diagonal: bool = False,
    ) -> Structure:
        """Generate the supercell for a defect.

        Args:
            sc_mat: supercell matrix if None, the supercell will be determined by `CubicSupercellAnalyzer`.
            dummy_species: Dummy species to highlight the defect position (for visualizing vacancies).
            max_atoms: Maximum number of atoms allowed in the supercell.
            min_atoms: Minimum number of atoms allowed in the supercell.
            min_length: Minimum length of the smallest supercell lattice vector.
            force_diagonal: If True, return a transformation with a diagonal transformation matrix.

        Returns:
            Structure: The supercell structure.
        """
        if sc_mat is None:
            sc_mat = get_sc_fromstruct(
                self.structure,
                min_atoms=min_atoms,
                max_atoms=max_atoms,
                min_length=min_length,
                force_diagonal=force_diagonal,
            )

        sc_structure = self.structure * sc_mat
        sc_mat_inv = np.linalg.inv(sc_mat)
        sc_pos = np.dot(self.site.frac_coords, sc_mat_inv)
        sc_site = PeriodicSite(self.site.specie, sc_pos, sc_structure.lattice)

        sc_defect = self.__class__(
            structure=sc_structure, site=sc_site, oxi_state=self.oxi_state
        )
        sc_defect_struct = sc_defect.defect_structure
        sc_defect_struct.remove_oxidation_states()
        if dummy_species is not None:
            dummy_pos = np.dot(self.site.frac_coords, sc_mat_inv)
            dummy_pos = np.mod(dummy_pos, 1)
            sc_defect_struct.insert(len(sc_structure), dummy_species, dummy_pos)

        return sc_defect_struct

    @property
    def symmetrized_structure(self) -> SymmetrizedStructure:
        """Returns the multiplicity of a defect site within the structure.

        This is required for concentration analysis and confirms that defect_site is a
        site in bulk_structure.
        """
        sga = SpacegroupAnalyzer(
            self.structure, symprec=self.symprec, angle_tolerance=self.angle_tolerance
        )
        return sga.get_symmetrized_structure()

    def __eq__(self, __o: object) -> bool:
        """Equality operator."""
        if not isinstance(__o, Defect):
            raise TypeError("Can only compare Defects to Defects")
        sm = StructureMatcher(comparator=ElementComparator())
        return sm.fit(self.defect_structure, __o.defect_structure)

    @property
    def defect_type(self) -> int:
        """Get the defect type.

        Returns:
            int: The defect type.
        """
        return getattr(DefectType, self.__class__.__name__)


class Vacancy(Defect):
    """Class representing a vacancy defect."""

    def get_multiplicity(self) -> int:
        """Returns the multiplicity of a defect site within the structure.

        This is required for concentration analysis and confirms that defect_site is
        a site in bulk_structure.

        Returns:
            int: The multiplicity of the defect.
        """
        symm_struct = self.symmetrized_structure
        defect_site = self.structure[self.defect_site_index]
        equivalent_sites = symm_struct.find_equivalent_sites(defect_site)
        return len(equivalent_sites)

    @property
    def name(self) -> str:
        """Name of the defect."""
        return f"v_{get_element(self.defect_site.specie)}"

    @property
    def defect_site(self):
        """Returns the site in the structure that corresponds to the defect site."""
        res = min(
            self.structure.get_sites_in_sphere(
                self.site.coords, 0.1, include_index=True
            ),
            key=lambda x: x[1],
        )
        return res

    @property
    def defect_site_index(self) -> int:
        """Get the index of the defect in the structure."""
        return self.defect_site.index

    @property
    def defect_structure(self):
        """Returns the defect structure with the proper oxidation state."""
        struct = self.structure.copy()
        struct.remove_sites([self.defect_site_index])
        return struct

    @property
    def element_changes(self) -> Dict[Element, int]:
        """Get the species changes of the vacancy defect.

        Returns:
            Dict[Element, int]: The species changes of the defect.
        """
        return {self.structure.sites[self.defect_site_index].specie.element: -1}

    def _guess_oxi_state(self) -> float:
        """Best guess for the oxidation state of the defect.

        For vacancies, the oxidation state is the opposite of the oxidation state of the
        removed atom.

        Returns:
            float: The oxidation state of the defect.
        """
        return -self.defect_site.specie.oxi_state

    def __repr__(self) -> str:
        """Representation of a vacancy defect."""
        vac_species = get_element(self.defect_site.specie)
        return f"{vac_species} Vacancy defect at site #{self.defect_site_index}"


class Substitution(Defect):
    """Single-site substitutional defects."""

    def __init__(
        self,
        structure: Structure,
        site: PeriodicSite,
        multiplicity: int | None = None,
        oxi_state: float | None = None,
        **kwargs,
    ) -> None:
        """Initialize a substitutional defect object.

        The position of `site` determines the atom to be removed and the species of
        `site` determines the replacing species.

        Args:
            structure: The structure of the defect.
            site: Replace the nearest site with this one.
            multiplicity: The multiplicity of the defect.
            oxi_state: The oxidation state of the defect, if not specified,
            this will be determined automatically.
        """
        super().__init__(structure, site, multiplicity, oxi_state, **kwargs)

    def get_multiplicity(self) -> int:
        """Returns the multiplicity of a defect site within the structure.

        This is required for concentration analysis and confirms that defect_site is
        a site in bulk_structure.
        """
        symm_struct = self.symmetrized_structure
        defect_site = self.structure[self.defect_site_index]
        equivalent_sites = symm_struct.find_equivalent_sites(defect_site)
        return len(equivalent_sites)

    @property
    def name(self) -> str:
        """Name of the defect."""
        return f"{get_element(self.site.specie)}_{get_element(self.defect_site.specie)}"

    @property
    def defect_structure(self) -> Structure:
        """Returns the defect structure."""
        struct: Structure = self.structure.copy()
        rm_oxi = struct.sites[self.defect_site_index].specie.oxi_state
        struct.remove_sites([self.defect_site_index])
        sub_states = self.site.specie.icsd_oxidation_states
        if len(sub_states) == 0:
            sub_states = self.site.specie.oxidation_states
        sub_oxi = min(sub_states, key=lambda x: abs(x - rm_oxi))
        sub_specie = Species(self.site.specie.symbol, sub_oxi)
        struct.insert(
            self.defect_site_index,
            species=sub_specie,
            coords=np.mod(self.site.frac_coords, 1),
        )
        return struct

    @property
    def defect_site(self):
        """Returns the site in the structure that corresponds to the defect site."""
        return min(
            self.structure.get_sites_in_sphere(
                self.site.coords, 0.1, include_index=True
            ),
            key=lambda x: x[1],
        )

    @property
    def defect_site_index(self) -> int:
        """Get the index of the defect in the structure."""
        return self.defect_site.index

    @property
    def element_changes(self) -> Dict[Element, int]:
        """Get the species changes of the substitution defect.

        Returns:
            Dict[Element, int]: The species changes of the defect.
        """
        return {
            self.structure.sites[self.defect_site_index].specie.element: -1,
            self.site.specie.element: +1,
        }

    def _guess_oxi_state(self) -> float:
        """Best guess for the oxidation state of the defect.

        For a substitution defect, the oxidation state of the defect is given
        by the difference between the oxidation state of the new and old atoms.

        Returns:
            float: The oxidation state of the defect.
        """
        orig_site = self.defect_site
        sub_site = self.defect_structure[self.defect_site_index]
        return sub_site.specie.oxi_state - orig_site.specie.oxi_state

    def __repr__(self) -> str:
        """Representation of a substitutional defect."""
        rm_species = get_element(self.defect_site.specie)
        sub_species = get_element(self.site.specie)
        return (
            f"{sub_species} subsitituted on the {rm_species} site at "
            f"at site #{self.defect_site_index}"
        )


class Interstitial(Defect):
    """Interstitial Defect."""

    def __init__(
        self,
        structure: Structure,
        site: PeriodicSite,
        multiplicity: int = 1,
        oxi_state: float | None = None,
        **kwargs,
    ) -> None:
        """Initialize an interstitial defect object.

        The interstitial defect effectively inserts the `site` object into the structure.

        Args:
            structure: The structure of the defect.
            site: Inserted site, also determines the species.
            multiplicity: The multiplicity of the defect.
            oxi_state: The oxidation state of the defect, if not specified,
                this will be determined automatically.
        """
        super().__init__(structure, site, multiplicity, oxi_state, **kwargs)

    def get_multiplicity(self) -> int:
        """Determine the multiplicity of the defect site within the structure."""
        raise NotImplementedError(
            "Interstitial multiplicity should be determined by the generator."
        )

    @property
    def name(self) -> str:
        """Name of the defect."""
        return f"{get_element(self.site.specie)}_i"

    @property
    def defect_structure(self) -> Structure:
        """Returns the defect structure."""
        struct: Structure = self.structure.copy()
        # use the highest value oxidation state among the two most popular ones
        # found in the ICSD
        inter_states = self.site.specie.icsd_oxidation_states[:2]
        if len(inter_states) == 0:
            _logger.warning(
                f"No oxidation states found for {self.site.specie.symbol}. "
                "in ICSD using `oxidation_states` without frequencuy ranking."
            )
            inter_states = self.site.specie.oxidation_states
        inter_oxi = max(inter_states, key=abs)
        int_specie = Species(self.site.specie.symbol, inter_oxi)
        struct.insert(
            0,
            species=int_specie,
            coords=np.mod(self.site.frac_coords, 1),
        )
        return struct

    @property
    def defect_site_index(self) -> int:
        """Get the index of the defect in the structure."""
        return 0

    @property
    def element_changes(self) -> Dict[Element, int]:
        """Get the species changes of the intersitial defect.

        Returns:
            Dict[Element, int]: The species changes of the defect.
        """
        return {
            self.site.specie.element: +1,
        }

    def _guess_oxi_state(self) -> float:
        """Best guess for the oxidation state of the defect.

        For interstitials, just use the oxidation state of the site.
        The oxidation of the interstitial site is determined by highest
        absolute value of the oxidation states of the inserted atom.

        Returns:
            float: The oxidation state of the defect.
        """
        sub_site = self.defect_structure[self.defect_site_index]
        return sub_site.specie.oxi_state

    def __repr__(self) -> str:
        """Representation of a interstitial defect."""
        sub_species = get_element(self.site.specie)
        fpos_str = ",".join(f"{x:.2f}" for x in self.site.frac_coords)
        return f"{sub_species} intersitial site at [{fpos_str}]"


class DefectComplex(Defect):
    """A complex of defects."""

    def __init__(
        self,
        defects: list[Defect],
        oxi_state: float | None = None,
    ) -> None:
        """Initialize a complex defect object.

        Args:
            defects: List of defects.
            oxi_state: The oxidation state of the defect, if not specified,
                this will be determined automatically.
        """
        self.defects = defects
        self.structure = self.defects[0].structure
        self.oxi_state = self._guess_oxi_state() if oxi_state is None else oxi_state

    def __repr__(self) -> str:
        """Representation of a complex defect."""
        return f"Complex defect containing: {[d.name for d in self.defects]}"

    def get_multiplicity(self) -> int:
        """Determine the multiplicity of the defect site within the structure."""
        raise NotImplementedError("Not implemented for defect complexes")

    @property
    def element_changes(self) -> Dict[Element, int]:
        """Determine the species changes of the complex defect."""
        cnt: dict[Element, int] = collections.defaultdict(int)
        for defect in self.defects:
            for el, change in defect.element_changes.items():
                cnt[el] += change
        return dict(cnt)

    @property
    def name(self) -> str:
        """Name of the defect."""
        return "+".join([d.name for d in self.defects])

    def _guess_oxi_state(self) -> float:
        oxi_state = 0.0
        for defect in self.defects:
            oxi_state += defect.oxi_state
        return oxi_state

    @property
    def defect_structure(self) -> Structure:
        """Returns the defect structure."""
        defect_structure = self.structure.copy()
        for defect in self.defects:
            update_structure(
                defect_structure, defect.site, defect_type=defect.defect_type
            )
        return defect_structure

    def get_supercell_structure(
        self,
        sc_mat: np.ndarray | None = None,
        dummy_species: Species | None = None,
        min_atoms: int = 80,
        max_atoms: int = 240,
        min_length: float = 10.0,
        force_diagonal: bool = False,
    ) -> Structure:
        """Generate the supercell for a defect.

        Args:
            sc_mat: supercell matrix if None, the supercell will be determined by `CubicSupercellAnalyzer`.
            dummy_species: Dummy species used for visualization. Will be placed at the average
                position of the defect sites.
            max_atoms: Maximum number of atoms allowed in the supercell.
            min_atoms: Minimum number of atoms allowed in the supercell.
            min_length: Minimum length of the smallest supercell lattice vector.
            force_diagonal: If True, return a transformation with a diagonal transformation matrix.

        Returns:
            Structure: The supercell structure.
        """
        if sc_mat is None:
            sc_mat = get_sc_fromstruct(
                self.structure,
                min_atoms=min_atoms,
                max_atoms=max_atoms,
                min_length=min_length,
                force_diagonal=force_diagonal,
            )
        sc_structure = self.structure * sc_mat
        sc_mat_inv = np.linalg.inv(sc_mat)
        complex_pos = np.zeros(3)
        for defect in self.defects:
            sc_pos = np.dot(defect.site.frac_coords, sc_mat_inv)
            complex_pos += sc_pos
            sc_site = PeriodicSite(defect.site.specie, sc_pos, sc_structure.lattice)
            update_structure(sc_structure, sc_site, defect_type=defect.defect_type)
        complex_pos /= len(self.defects)
        if dummy_species is not None:
            for defect in self.defects:
                dummy_pos = np.dot(defect.site.frac_coords, sc_mat_inv)
                dummy_pos = np.mod(dummy_pos, 1)
                sc_structure.insert(len(sc_structure), dummy_species, dummy_pos)

        return sc_structure


def update_structure(structure, site, defect_type):
    """Update the structure with the defect site.

    Types of operations:
        1. Vacancy: remove the site.
        2. Substitution: replace the site with the defect species.
        3. Interstitial: insert the defect species at the site.

    Args:
        structure: The structure to be updated.
        site: The defect site.
        defect_type: The type of the defect.

    Returns:
        Structure: The updated structure.
    """

    def _update(structure, site, rm: bool, replace: bool):
        in_sphere = structure.get_sites_in_sphere(site.coords, 0.1, include_index=True)

        if len(in_sphere) == 0 and rm:  # pragma: no cover
            raise ValueError("No site found to remove.")

        if rm or replace:
            rm_site = min(
                in_sphere,
                key=lambda x: x[1],
            )
            rm_index = rm_site.index
            structure.remove_sites([rm_index])

        if rm:
            return

        sub_specie = Element(site.specie.symbol)
        structure.insert(
            0,
            species=sub_specie,
            coords=site.frac_coords,
        )

    if defect_type == DefectType.Vacancy:
        _update(structure, site, rm=True, replace=False)
    elif defect_type == DefectType.Substitution:
        _update(structure, site, rm=False, replace=True)
    elif defect_type == DefectType.Interstitial:
        _update(structure, site, rm=False, replace=False)
    else:
        raise ValueError("Unknown point defect type.")


class Adsorbate(Interstitial):
    """Subclass of Interstitial with a different name.

    Used for keeping track of adsorbate, which are treated the same
    algorithmically as interstitials, but are conceptually separate.
    """

    @property
    def name(self) -> str:
        """Returns a name for this defect."""
        return f"{get_element(self.site.specie)}_{{ads}}"

    def __repr__(self) -> str:
        """Representation of a adsorbate defect."""
        sub_species = get_element(self.site.specie)
        fpos_str = ",".join(f"{x:.2f}" for x in self.site.frac_coords)
        return f"{sub_species} adsorbate site at [{fpos_str}]"


def get_element(sp_el: Species | Element) -> Element:
    """Get the element from a species or element."""
    if isinstance(sp_el, Species):
        return sp_el.element
    return sp_el


def get_vacancy(structure: Structure, isite: int, **kwargs) -> Vacancy:
    """Get a vacancy defect from a structure and site index.

    Convenience function for creating a Vacancy object quickly.

    Args:
        structure: The structure to create the vacancy in.
        isite: The site index of the vacancy.
        **kwargs: Keyword arguments to pass to Vacancy constructor.
    """
    site = structure[isite]
    return Vacancy(structure=structure, site=site, **kwargs)


# TODO: matching defect complexes might be done with some kind of CoM site to fix the periodicity
# Get this by taking the periodic average of all the provided sites.
# class DefectComplex(DummySpecies):
#     def __init__(self, oxidation_state: float = 0, properties: dict | None = None):
#         super().__init__("Vac", oxidation_state, properties)
