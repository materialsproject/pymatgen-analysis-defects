"""Classes representing defects."""

from __future__ import annotations

import collections
import logging
from abc import ABCMeta, abstractmethod, abstractproperty
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from monty.json import MSONable
from pymatgen.analysis.defects.supercells import get_sc_fromstruct
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import Element, PeriodicSite, Species
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .utils import get_plane_spacing

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure
    from pymatgen.symmetry.structure import SymmetrizedStructure
    from typing_extensions import Self

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

RNG = np.random.default_rng(42)


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
        equivalent_sites: list[PeriodicSite] | None = None,
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
            equivalent_sites: A list of equivalent sites for the defect in the structure.
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
        self.equivalent_sites = equivalent_sites if equivalent_sites is not None else []
        self.user_charges = user_charges if user_charges else []
        if oxi_state is None:
            # Try to use the reduced cell first since oxidation state assignment
            # scales poorly with systems size.
            try:
                self.structure.add_oxidation_state_by_guess(max_sites=-1)
                # check oxi_states assigned and not all zero
                if all(specie.oxi_state == 0 for specie in self.structure.species):
                    self.structure.add_oxidation_state_by_guess()
            except Exception:  # noqa: BLE001 # pragma: no cover
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
    def element_changes(self) -> dict[Element, int]:
        """Get the species changes of the defect.

        Returns:
            Dict[Element, int]: The species changes of the defect.
        """

    @property
    def centered_defect_structure(self) -> Structure:
        """Get a defect structure that is centered around the site.

        Move all the sites in the structure so that they
        are in the periodic image closest to the defect site.
        """
        return center_structure(self.defect_structure, self.site.frac_coords)

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
        else:  # pragma: no cover
            sign = -1 if self.oxi_state < 0 else 1
            oxi_state = sign * int(np.ceil(abs(self.oxi_state)))
            _logger.warning(
                "Non-integer oxidation state detected."
                "Round to integer with larger absolute value: %s -> %s",
                self.oxi_state,
                oxi_state,
            )

        if oxi_state >= 0:
            charges = [*range(-padding, oxi_state + padding + 1)]
        else:
            charges = [*range(oxi_state - padding, padding + 1)]

        return charges

    def get_supercell_structure(
        self,
        sc_mat: np.ndarray | None = None,
        defect_structure: Structure | None = None,
        dummy_species: str | None = None,
        min_atoms: int = 80,
        max_atoms: int = 240,
        min_length: float = 10.0,
        force_diagonal: bool = False,
        relax_radius: float | str | None = None,
        perturb: float | None = None,
        target_frac_coords: np.ndarray | None = None,
        return_site: bool = False,
    ) -> Structure:
        """Generate the supercell for a defect.

        If the bulk structure (provided by `Defect.structure`) and defect structures
        have oxidation state information, then the supercell will be decorated with
        oxidation states.  Otherwise, the supercell structure will not have any oxidation
        state information.  This also allows for oxidation state decoration of different
        defect charge states.

        .. code-block:: python
            defect_struct = defect.defect_structure.copy()
            defect_struct.add_oxidation_state_by_guess(target_charge=2)

        .. note::
            Since any algorithm for decorating the oxidation states will be combinatorial,
            They can be very slow for large supercells. If you want to decorate the structure
            with oxidation states, you have to first decorate the smaller unit cell defect
            structure then implant it in the supercell.

        Args:
            sc_mat: supercell matrix if None, the supercell will be determined by `CubicSupercellAnalyzer`.
            defect_structure: Alternative defect structure to use for generating the supercell.
                You might want to use this if you want to decorate the oxidation states of the
                defect structure using a custom method.
            dummy_species: Dummy species to highlight the defect position (for visualizing vacancies).
            max_atoms: Maximum number of atoms allowed in the supercell.
            min_atoms: Minimum number of atoms allowed in the supercell.
            min_length: Minimum length of the smallest supercell lattice vector.
            force_diagonal: If True, return a transformation with a diagonal transformation matrix.
            relax_radius: Relax the supercell atoms to a sphere of this radius around the defect site.
            perturb: The amount to perturb the sites in the supercell. Only perturb the sites with
                selective dynamics set to True. So this setting only works with `relax_radius`.
            target_frac_coords: If set, defect will be placed at the closest equivalent site to these
                fractional coordinates.
            return_site: If True, returns a tuple of the (defect supercell, defect site position).

        Returns:
            Structure: The supercell structure.
            PeriodicSite (optional): The position of the defect site in the supercell.
        """

        def _has_oxi(struct: Structure) -> bool:
            return all(hasattr(site.specie, "oxi_state") for site in struct)

        if defect_structure is None:
            defect_structure = self.centered_defect_structure
        bulk_structure = center_structure(self.structure, self.site.frac_coords)
        keep_oxi = _has_oxi(bulk_structure) and _has_oxi(defect_structure)

        if sc_mat is None:
            sc_mat = get_sc_fromstruct(
                bulk_structure,
                min_atoms=min_atoms,
                max_atoms=max_atoms,
                min_length=min_length,
                force_diagonal=force_diagonal,
            )

        # Get the translation vector in to the desired cell in Cartesian coordinates
        if target_frac_coords is None:
            target_frac_coords = [1e-6, 1e-6, 1e-6]
        trans_R = np.floor(np.dot(target_frac_coords, sc_mat))
        trans_vec = np.dot(trans_R, bulk_structure.lattice.matrix)
        sc_defect_struct = bulk_structure * sc_mat

        # Use the site as a reference point for the position
        sc_mat_inv = np.linalg.inv(sc_mat)
        sc_pos = np.dot(self.site.frac_coords, sc_mat_inv)
        sc_site = PeriodicSite(
            species=self.site.specie,
            coords=sc_pos,
            lattice=sc_defect_struct.lattice,
            coords_are_cartesian=False,
        )

        bulk_site_mapping = _get_mapped_sites(bulk_structure, sc_defect_struct)
        defect_site_mapping = _get_mapped_sites(defect_structure, sc_defect_struct)

        # Remove the indices that that are not mapped
        rm_indices = set(bulk_site_mapping.values()) - set(defect_site_mapping.values())

        # Set the species for the sites that are mapped
        for defect_site, bulk_site in defect_site_mapping.items():
            sc_defect_struct[bulk_site]._species = defect_structure[defect_site].species

        # interstitials
        int_uc_indices = set(range(len(defect_structure))) - set(
            defect_site_mapping.keys(),
        )
        for i in int_uc_indices:
            int_sc_pos = np.dot(defect_structure[i].frac_coords, sc_mat_inv)
            sc_defect_struct.insert(
                0,
                defect_structure[i].specie,
                int_sc_pos,
                coords_are_cartesian=False,
            )

        # Remove the sites that are not mapped
        sc_defect_struct.remove_sites(list(rm_indices))
        _set_selective_dynamics(
            structure=sc_defect_struct,
            site_pos=sc_site.coords,
            relax_radius=relax_radius,
        )
        if perturb is not None:
            _perturb_dynamic_sites(sc_defect_struct, distance=perturb)

        # Translate the structure to the target position
        sc_defect_struct.translate_sites(
            indices=list(range(len(sc_defect_struct))),
            vector=trans_vec,
            frac_coords=False,
        )
        sc_site._frac_coords += trans_R

        if not keep_oxi:
            sc_defect_struct.remove_oxidation_states()

        if dummy_species is not None:
            sc_defect_struct.insert(0, dummy_species, sc_site.frac_coords)

        if return_site:
            return sc_defect_struct, sc_site
        return sc_defect_struct

    @property
    def symmetrized_structure(self) -> SymmetrizedStructure:
        """Get the symmetrized version of the bulk structure."""
        sga = SpacegroupAnalyzer(
            self.structure,
            symprec=self.symprec,
            angle_tolerance=self.angle_tolerance,
        )
        return sga.get_symmetrized_structure()

    def __eq__(self, __o: object) -> bool:
        """Equality operator."""
        if not isinstance(__o, Defect):  # pragma: no cover
            msg = "Can only compare Defects to Defects"
            raise TypeError(msg)
        sm = StructureMatcher(comparator=ElementComparator())
        return sm.fit(self.defect_structure, __o.defect_structure)

    @property
    def defect_type(self) -> DefectType:
        """Get the defect type.

        Returns:
            int: The defect type.
        """
        return getattr(DefectType, self.__class__.__name__)

    @property
    def latex_name(self) -> str:
        """Get the latex name of the defect.

        Returns:
            str: The latex name of the defect.
        """
        root, suffix = self.name.split("_")
        return rf"{root}$_{{\rm {suffix}}}$"


class NamedDefect(MSONable):
    """Class for defect definition without the UC structure.

    The defect is defined only by its name. For complexes the name
    should be created with "+" between the individual defects.

    .. note::
        This class is only used to help aggregate defects calculated
        outside of our framework so a ``Defect`` object is missing.
        The object will not have any mechanism for generating supercells
        or guessing oxidation states.  It is simply a placeholder to help
        with the grouping logic of the Formation Energy diagram analysis.

    """

    def __init__(self, name: str, bulk_formula: str, element_changes: dict) -> None:
        """Initialize a NamedDefect object.

        Args:
            name: The name of the defect.
            bulk_formula: The formula of the bulk structure.
            element_changes: The species changes of the defect.
        """
        self.name = name
        self.bulk_formula = bulk_formula
        self.element_changes = element_changes

    @classmethod
    def from_structures(
        cls, defect_structure: Structure, bulk_structure: Structure
    ) -> Self:
        """Initialize a NameDefect object from structures.

        Args:
            defect_structure: The structure of the defect.
            bulk_structure: The structure of the bulk.

        Returns:
            NamedDefect: The defect object.
        """
        el_diff = _get_el_changes_from_structures(defect_structure, bulk_structure)
        name_ = _get_defect_name(el_diff)
        bulk_formula = bulk_structure.composition.reduced_formula
        return cls(name=name_, bulk_formula=bulk_formula, element_changes=el_diff)

    @property
    def latex_name(self) -> str:
        """Get the latex name of the defect.

        Returns:
            str: The latex name of the defect.
        """
        single_names = self.name.split("+")
        l_names = []
        for n in single_names:
            root, suffix = n.split("_")
            l_names.append(rf"{root}$_{{\rm {suffix}}}$")
        return " + ".join(l_names)

    def __eq__(self, __value: object) -> bool:
        """Only need to compare names."""
        if not isinstance(__value, NamedDefect):  # pragma: no cover
            msg = "Can only compare NamedDefects to NamedDefects"
            raise TypeError(msg)
        return self.__repr__() == __value.__repr__()

    def __repr__(self) -> str:
        """String representation of the NamedDefect."""
        return f'{self.bulk_formula}:{"+".join(sorted(self.name.split("+")))}'


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
    def defect_site(self) -> PeriodicSite:
        """Returns the site in the structure that corresponds to the defect site."""
        return min(
            self.structure.get_sites_in_sphere(
                self.site.coords,
                0.1,
                include_index=True,
            ),
            key=lambda x: x[1],
        )

    @property
    def defect_site_index(self) -> int:
        """Get the index of the defect in the structure."""
        return self.defect_site.index

    @property
    def defect_structure(self) -> Structure:
        """Returns the defect structure with the proper oxidation state."""
        struct = self.structure.copy()
        struct.remove_sites([self.defect_site_index])
        return struct

    @property
    def element_changes(self) -> dict[Element, int]:
        """Get the species changes of the vacancy defect.

        Returns:
            Dict[Element, int]: The species changes of the defect.
        """
        return {get_element(self.structure.sites[self.defect_site_index].specie): -1}

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
        equivalent_sites: list[PeriodicSite] | None = None,
        symprec: float = 0.01,
        angle_tolerance: float = 5,
        user_charges: list[int] | None = None,
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
            equivalent_sites: A list of equivalent sites for the defect in the structure.
            symprec: Tolerance for symmetry finding.
            angle_tolerance: Angle tolerance for symmetry finding.
            user_charges: User specified charge states. If specified,
                ``get_charge_states`` will return this list. If ``None`` or empty list
                the charge states will be determined automatically.
        """
        super().__init__(
            structure=structure,
            site=site,
            multiplicity=multiplicity,
            oxi_state=oxi_state,
            equivalent_sites=equivalent_sites,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            user_charges=user_charges,
        )

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
        struct.remove_sites([self.defect_site_index])
        insert_el = get_element(self.site.specie)
        struct.insert(
            self.defect_site_index,
            species=insert_el,
            coords=np.mod(self.site.frac_coords, 1),
        )
        return struct

    @property
    def defect_site(self) -> PeriodicSite:
        """Returns the site in the structure that corresponds to the defect site."""
        return min(
            self.structure.get_sites_in_sphere(
                self.site.coords,
                0.1,
                include_index=True,
            ),
            key=lambda x: x[1],
        )

    @property
    def defect_site_index(self) -> int:
        """Get the index of the defect in the structure."""
        return self.defect_site.index

    @property
    def element_changes(self) -> dict[Element, int]:
        """Get the species changes of the substitution defect.

        Returns:
            Dict[Element, int]: The species changes of the defect.
        """
        return {
            get_element(self.structure.sites[self.defect_site_index].specie): -1,
            get_element(self.site.specie): +1,
        }

    def _guess_oxi_state(self) -> float:
        """Best guess for the oxidation state of the defect.

        For a substitution defect, the oxidation state of the defect is given
        by the difference between the oxidation state of the new and old atoms.

        Returns:
            float: The oxidation state of the defect.
        """
        rm_oxi = self.structure[self.defect_site_index].specie.oxi_state

        # check if substitution atom is present in structure (i.e. antisite substitution):
        sub_elt_sites_in_struct = [
            site
            for site in self.structure
            if site.specie.symbol
            == self.site.specie.symbol  # gives Element symbol (without oxi state)
        ]
        if len(sub_elt_sites_in_struct) == 0:
            sub_states = self.site.specie.common_oxidation_states
            if len(sub_states) == 0:  # pragma: no cover
                msg = (
                    f"No common oxidation states found for {self.site.specie}."
                    "Please specify the oxidation state manually."
                )
                raise ValueError(
                    msg,
                )
            sub_oxi = min(sub_states, key=lambda x: abs(x - rm_oxi))
        else:
            sub_oxi = int(
                np.mean([site.specie.oxi_state for site in sub_elt_sites_in_struct]),
            )

        return sub_oxi - rm_oxi

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
        equivalent_sites: list[PeriodicSite] | None = None,
        symprec: float = 0.01,
        angle_tolerance: float = 5,
        user_charges: list[int] | None = None,
    ) -> None:
        """Initialize an interstitial defect object.

        The interstitial defect effectively inserts the `site` object into the structure.

        Args:
            structure: The structure of the defect.
            site: Inserted site, also determines the species.
            multiplicity: The multiplicity of the defect.
            oxi_state: The oxidation state of the defect, if not specified,
                this will be determined automatically.
            equivalent_sites: A list of equivalent sites for the defect in the structure.
            symprec: Tolerance for symmetry finding.
            angle_tolerance: Angle tolerance for symmetry finding.
            user_charges: User specified charge states. If specified,
        """
        super().__init__(
            structure=structure,
            site=site,
            multiplicity=multiplicity,
            oxi_state=oxi_state,
            equivalent_sites=equivalent_sites,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            user_charges=user_charges,
        )

    def get_multiplicity(self) -> int:
        """Determine the multiplicity of the defect site within the structure."""
        msg = "Interstitial multiplicity should be determined by the generator."
        raise NotImplementedError(
            msg,
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
                "No oxidation states found for %s. "
                "in ICSD using `oxidation_states` without frequency ranking.",
                self.site.specie.symbol,
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
    def element_changes(self) -> dict[Element, int]:
        """Get the species changes of the intersitial defect.

        Returns:
            Dict[Element, int]: The species changes of the defect.
        """
        return {
            get_element(self.site.specie): +1,
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
        defect_sites = [d.site for d in self.defects]
        center_of_mass = np.mean([s.coords for s in defect_sites], axis=0)
        self.site = PeriodicSite(
            species=DummySpecies(),
            coords=center_of_mass,
            lattice=self.structure.lattice,
            coords_are_cartesian=True,
        )

    def __repr__(self) -> str:
        """Representation of a complex defect."""
        return f"Complex defect containing: {[d.name for d in self.defects]}"

    def __eq__(self, __o: object) -> bool:
        """Check if  are equal."""
        if not isinstance(__o, Defect):
            msg = "Can only compare Defects to Defects"
            raise TypeError(msg)
        sm = StructureMatcher(comparator=ElementComparator())
        this_structure = self.defect_structure_with_com
        if isinstance(__o, DefectComplex):
            that_structure = __o.defect_structure_with_com
        else:
            that_structure = __o.defect_structure
        return sm.fit(this_structure, that_structure)

    @property
    def defect_structure_with_com(self) -> Structure:
        """Returns the defect structure with the center of mass as dummy site."""
        struct = self.defect_structure.copy()
        struct.insert(0, self.site.specie, self.site.frac_coords)
        return struct

    def get_multiplicity(self) -> int:
        """Determine the multiplicity of the defect site within the structure."""
        msg = "Not implemented for defect complexes"
        raise NotImplementedError(
            msg,
        )  # pragma: no cover

    @property
    def element_changes(self) -> dict[Element, int]:
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
                defect_structure,
                defect.site,
                defect_type=defect.defect_type,
            )
        return defect_structure

    @property
    def latex_name(self) -> str:
        """Get the latex name of the defect."""
        single_names = [d.latex_name for d in self.defects]
        return "$+$".join(single_names)


def update_structure(
    structure: Structure, site: PeriodicSite, defect_type: DefectType
) -> None:
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

    def _update(
        structure: Structure, site: PeriodicSite, rm: bool, replace: bool
    ) -> None:
        in_sphere = structure.get_sites_in_sphere(site.coords, 0.1, include_index=True)

        if len(in_sphere) == 0 and rm:  # pragma: no cover
            msg = "No site found to remove."
            raise ValueError(msg)

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
        msg = "Unknown point defect type."
        raise ValueError(msg)  # pragma: no cover


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


def _set_selective_dynamics(
    structure: Structure,
    site_pos: ArrayLike,
    relax_radius: float | str | None,
) -> None:
    """Set the selective dynamics behavior.

    Allow atoms to move for sites within a given radius of a given site,
    all other atoms are fixed.  Modify the structure in place.

    Args:
        structure: The structure to set the selective dynamics.
        site_pos: The center of the relaxation sphere.
        relax_radius: The radius of the relaxation sphere.
    """
    if relax_radius is None:
        return
    if relax_radius == "auto":
        relax_radius = min(get_plane_spacing(structure.lattice.matrix)) / 2.0
    if not isinstance(relax_radius, float):
        msg = "relax_radius must be a float or 'auto' or None"
        raise ValueError(msg)
    structure.get_sites_in_sphere(site_pos, relax_radius)
    relax_sites = structure.get_sites_in_sphere(
        site_pos,
        relax_radius,
        include_index=True,
    )
    relax_indices = [site.index for site in relax_sites]
    relax_mask = [[False, False, False]] * len(structure)
    for i in relax_indices:
        relax_mask[i] = [True, True, True]
    structure.add_site_property("selective_dynamics", relax_mask)


def perturb_sites(
    structure: Structure,
    distance: float,
    min_distance: float | None = None,
    site_indices: list | None = None,
) -> None:
    """Performs a random perturbation.

    Perturb the sites in a structure to break symmetry.  This is useful for
    finding energy minimum configurations.

    Args:
        structure (Structure): Input structure.
        distance (float): Distance in angstroms by which to perturb each
            site.
        min_distance (None, int, or float): if None, all displacements will
            be equal amplitude. If int or float, perturb each site a
            distance drawn from the uniform distribution between
            'min_distance' and 'distance'.
        site_indices (list): List of site indices on which to perform the
            perturbation. If None, all sites will be perturbed.

    """

    def get_rand_vec() -> ArrayLike:
        # deals with zero vectors.
        vector = RNG.normal(size=3)
        vnorm = np.linalg.norm(vector)
        dist = distance
        if isinstance(min_distance, (float, int)):
            dist = RNG.uniform(min_distance, dist)
        return vector / vnorm * dist if vnorm != 0 else get_rand_vec()

    if site_indices is None:
        site_indices_ = list(range(len(structure._sites)))
    else:
        site_indices_ = site_indices

    for i in site_indices_:
        structure.translate_sites([i], get_rand_vec(), frac_coords=False)


def _perturb_dynamic_sites(structure: Structure, distance: float) -> None:
    free_indices = [
        i
        for i, site in enumerate(structure)
        if site.properties["selective_dynamics"][0]
    ]
    perturb_sites(structure=structure, distance=distance, site_indices=free_indices)


def _get_mapped_sites(
    uc_structure: Structure, sc_structure: Structure, r: float = 0.001
) -> dict:
    """Get the list of sites indices in the supercell corresponding to the unit cell."""
    mapped_site_indices = {}
    for isite, uc_site in enumerate(uc_structure):
        sc_sites = sc_structure.get_sites_in_sphere(uc_site.coords, r)
        if len(sc_sites) == 1:
            mapped_site_indices[isite] = sc_sites[0].index
    return mapped_site_indices


def center_structure(structure: Structure, ref_fpos: ArrayLike) -> Structure:
    """Shift the sites around a center.

    Move all the sites in the structure so that they
    are in the periodic image closest to the reference fractional position.

    Args:
        structure: The structure to be centered.
        ref_fpos: The reference fractional position that will be set to the center.
    """
    struct = structure.copy()
    for idx, d_site in enumerate(struct):
        _, jiimage = struct.lattice.get_distance_and_image(ref_fpos, d_site.frac_coords)
        struct.translate_sites([idx], jiimage, to_unit_cell=False)
    return struct


def _get_el_changes_from_structures(defect_sc: Structure, bulk_sc: Structure) -> dict:
    """Get the name of the defect.

    Parse the defect structure and bulk structure to get the name of the defect.

    Args:
        defect_sc: The defect structure.
        bulk_sc: The bulk structure.

    Returns:
        dict: A dictionary representing the species changes in creating the defect.
    """

    def _check_int(n: float) -> bool:
        return isinstance(n, int) or n.is_integer()

    comp_defect = defect_sc.composition.element_composition
    comp_bulk = bulk_sc.composition.element_composition

    # get the element changes
    el_diff = {}
    for el, cnt in comp_defect.items():
        # has to be integer
        if not (_check_int(comp_bulk[el]) and _check_int(cnt)):
            msg = "Defect structure and bulk structure must have integer compositions."
            raise ValueError(
                msg,
            )
        tmp_ = int(cnt) - int(comp_bulk[el])
        if tmp_ != 0:
            el_diff[el] = tmp_
    return el_diff


def _get_defect_name(element_diff: dict) -> str:
    """Get the name of the defect.

    Parse the change in different elements to get the name of the defect.

    Args:
        element_diff: A dictionary representing the species changes of the defect.

    Returns:
        str: The name of the defect, if the defect is a complex, the names of the
            individual defects are separated by "+".
    """
    added_list = [(el, int(cnt)) for el, cnt in element_diff.items() if cnt > 0]
    removed_list = [(el, int(cnt)) for el, cnt in element_diff.items() if cnt < 0]

    # rank the elements by electronegativity
    added_list.sort(reverse=True)
    removed_list.sort(reverse=True)

    # get the different substitution names
    sub_names = []
    while added_list and removed_list:
        add_el, add_cnt = added_list.pop()
        rm_el, rm_cnt = removed_list.pop()
        sub_names.append(f"{add_el}_{rm_el}")
        add_cnt -= 1
        rm_cnt += 1
        if add_cnt != 0:
            added_list.append((add_el, add_cnt))
        if rm_cnt != 0:
            removed_list.append((rm_el, rm_cnt))

    # get the different vacancy names
    vac_names = []
    for el, _cnt in removed_list:
        vac_names.append(f"v_{el}")

    # get the different interstitial names
    int_names = []
    for el, _cnt in added_list:
        int_names.append(f"{el}_i")

    # combine the names
    return "+".join(sub_names + vac_names + int_names)
