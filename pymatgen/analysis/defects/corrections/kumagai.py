"""Kumagai defect correction module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats

from pymatgen.analysis.defects.core import DefectCorrection
from pymatgen.analysis.defects.utils import (
    generate_R_and_G_vecs,
    kumagai_to_V,
    tune_for_gamma,
)


class KumagaiCorrection(DefectCorrection):
    """
    A class for KumagaiCorrection class. Largely adapted from PyCDT code

    If this correction is used, please reference Kumagai and Oba's original paper
    (doi: 10.1103/PhysRevB.89.195205) as well as Freysoldt's original
    paper (doi: 10.1103/PhysRevLett.102.016402)

    NOTE that equations 8 and 9 from Kumagai et al. reference are divided by (4 pi) to get SI units
    """

    def __init__(self, dielectric_tensor, sampling_radius=None, gamma=None):
        """
        Initializes the Kumagai Correction
        Args:
            dielectric_tensor (float or 3x3 matrix): Dielectric constant for the structure

            optional data that can be tuned:
                sampling_radius (float): radius (in Angstrom) which sites must be outside
                    of to be included in the correction. Publication by Kumagai advises to
                    use Wigner-Seitz radius of defect supercell, so this is default value.
                gamma (float): convergence parameter for gamma function.
                    Code will automatically determine this if set to None.
        """
        self.metadata = {
            "gamma": gamma,
            "sampling_radius": sampling_radius,
            "potalign": None,
        }

        if isinstance(dielectric_tensor, (int, float)):
            self.dielectric = np.identity(3) * dielectric_tensor
        else:
            self.dielectric = np.array(dielectric_tensor)

    def get_correction(self, entry):
        """
        Gets the Kumagai correction for a defect entry
        Args:
            entry (DefectEntry): defect entry to compute Kumagai correction on.

                Requires following parameters in the DefectEntry to exist:

                    bulk_atomic_site_averages (list):  list of bulk structure"s atomic site averaged ESPs * charge,
                        in same order as indices of bulk structure
                        note this is list given by VASP's OUTCAR (so it is multiplied by a test charge of -1)

                    defect_atomic_site_averages (list):  list of defect structure"s atomic site averaged ESPs * charge,
                        in same order as indices of defect structure
                        note this is list given by VASP's OUTCAR (so it is multiplied by a test charge of -1)

                    site_matching_indices (list):  list of corresponding site index values for
                        bulk and defect site structures EXCLUDING the defect site itself
                        (ex. [[bulk structure site index, defect structure"s corresponding site index], ... ]

                    initial_defect_structure (Structure): Pymatgen Structure object representing un-relaxed defect
                        structure

                    defect_frac_sc_coords (array): Defect Position in fractional coordinates of the supercell
                        given in bulk_structure
        Returns:
            KumagaiCorrection values as a dictionary

        """
        bulk_atomic_site_averages = entry.parameters["bulk_atomic_site_averages"]
        defect_atomic_site_averages = entry.parameters["defect_atomic_site_averages"]
        site_matching_indices = entry.parameters["site_matching_indices"]
        defect_sc_structure = entry.parameters["initial_defect_structure"]
        defect_frac_sc_coords = entry.parameters["defect_frac_sc_coords"]

        lattice = defect_sc_structure.lattice
        volume = lattice.volume
        q = entry.defect.charge

        if not self.metadata["gamma"]:
            self.metadata["gamma"] = tune_for_gamma(lattice, self.dielectric)

        prec_set = [25, 28]
        g_vecs, recip_summation, r_vecs, real_summation = generate_R_and_G_vecs(
            self.metadata["gamma"], prec_set, lattice, self.dielectric
        )

        pot_shift = self.get_potential_shift(self.metadata["gamma"], volume)
        si = self.get_self_interaction(self.metadata["gamma"])
        es_corr = [
            (real_summation[ind] + recip_summation[ind] + pot_shift + si)
            for ind in range(2)
        ]

        # increase precision if correction is not converged yet
        # TODO: allow for larger prec_set to be tried if this fails
        if abs(es_corr[0] - es_corr[1]) > 0.0001:
            logger.debug(
                "Es_corr summation not converged! ({} vs. {})\nTrying a larger prec_set...".format(
                    es_corr[0], es_corr[1]
                )
            )
            prec_set = [30, 35]
            g_vecs, recip_summation, r_vecs, real_summation = generate_R_and_G_vecs(
                self.metadata["gamma"], prec_set, lattice, self.dielectric
            )
            es_corr = [
                (real_summation[ind] + recip_summation[ind] + pot_shift + si)
                for ind in range(2)
            ]
            if abs(es_corr[0] - es_corr[1]) < 0.0001:
                raise ValueError(
                    "Correction still not converged after trying prec_sets up to 35... serious error."
                )

        es_corr = es_corr[0] * -(q**2.0) * kumagai_to_V / 2.0  # [eV]

        # if no sampling radius specified for pot align, then assuming Wigner-Seitz radius:
        if not self.metadata["sampling_radius"]:
            wz = lattice.get_wigner_seitz_cell()
            dist = []
            for facet in wz:
                midpt = np.mean(np.array(facet), axis=0)
                dist.append(np.linalg.norm(midpt))
            self.metadata["sampling_radius"] = min(dist)

        # assemble site_list based on matching indices
        # [[defect_site object, Vqb for site], .. repeat for all non defective sites]
        site_list = []
        for bs_ind, ds_ind in site_matching_indices:
            Vqb = -(
                defect_atomic_site_averages[int(ds_ind)]
                - bulk_atomic_site_averages[int(bs_ind)]
            )
            site_list.append([defect_sc_structure[int(ds_ind)], Vqb])

        pot_corr = self.perform_pot_corr(
            defect_sc_structure,
            defect_frac_sc_coords,
            site_list,
            self.metadata["sampling_radius"],
            q,
            r_vecs[0],
            g_vecs[0],
            self.metadata["gamma"],
        )

        entry.parameters["kumagai_meta"] = dict(self.metadata)
        entry.parameters["potalign"] = pot_corr / (-q) if q else 0.0

        return {
            "kumagai_electrostatic": es_corr,
            "kumagai_potential_alignment": pot_corr,
        }

    def perform_pot_corr(
        self,
        defect_structure,
        defect_frac_coords,
        site_list,
        sampling_radius,
        q,
        r_vecs,
        g_vecs,
        gamma,
    ):
        """
        For performing potential alignment in manner described by Kumagai et al.
        Args:
            defect_structure: Pymatgen Structure object corresponding to the defect supercell

            defect_frac_coords (array): Defect Position in fractional coordinates of the supercell
                given in bulk_structure

            site_list: list of corresponding site index values for
                bulk and defect site structures EXCLUDING the defect site itself
                (ex. [[bulk structure site index, defect structure"s corresponding site index], ... ]

            sampling_radius (float): radius (in Angstrom) which sites must be outside
                of to be included in the correction. Publication by Kumagai advises to
                use Wigner-Seitz radius of defect supercell, so this is default value.

            q (int): Defect charge

            r_vecs: List of real lattice vectors to use in summation

            g_vecs: List of reciprocal lattice vectors to use in summation

            gamma (float): Ewald parameter

        Return:
            Potential alignment contribution to Kumagai Correction (float)
        """
        volume = defect_structure.lattice.volume
        potential_shift = self.get_potential_shift(gamma, volume)

        pot_dict = {}  # keys will be site index in the defect structure
        for_correction = []  # region to sample for correction

        # for each atom, do the following:
        # (a) get relative_vector from defect_site to site in defect_supercell structure
        # (b) recalculate the recip and real summation values based on this r_vec
        # (c) get information needed for pot align
        for site, Vqb in site_list:
            dist, jimage = site.distance_and_image_from_frac_coords(defect_frac_coords)
            vec_defect_to_site = defect_structure.lattice.get_cartesian_coords(
                site.frac_coords - jimage - defect_frac_coords
            )
            dist_to_defect = np.linalg.norm(vec_defect_to_site)
            if abs(dist_to_defect - dist) > 0.001:
                raise ValueError("Error in computing vector to defect")

            relative_real_vectors = [r_vec - vec_defect_to_site for r_vec in r_vecs[:]]

            real_sum = self.get_real_summation(gamma, relative_real_vectors)
            recip_sum = self.get_recip_summation(
                gamma, g_vecs, volume, r=vec_defect_to_site[:]
            )

            Vpc = (real_sum + recip_sum + potential_shift) * kumagai_to_V * q

            defect_struct_index = defect_structure.index(site)
            pot_dict[defect_struct_index] = {
                "Vpc": Vpc,
                "Vqb": Vqb,
                "dist_to_defect": dist_to_defect,
            }

            logger.debug(
                f"For atom {defect_struct_index}\n\tbulk/defect DFT potential difference = {Vqb}"
            )
            logger.debug(f"\tanisotropic model charge: {Vpc}")
            logger.debug(f"\t\treciprocal part: {recip_sum * kumagai_to_V * q}")
            logger.debug(f"\t\treal part: {real_sum * kumagai_to_V * q}")
            logger.debug(
                f"\t\tself interaction part: {potential_shift * kumagai_to_V * q}"
            )
            logger.debug(f"\trelative_vector to defect: {vec_defect_to_site}")

            if dist_to_defect > sampling_radius:
                logger.debug(
                    "\tdistance to defect is {} which is outside minimum sampling "
                    "radius {}".format(dist_to_defect, sampling_radius)
                )
                for_correction.append(Vqb - Vpc)
            else:
                logger.debug(
                    "\tdistance to defect is {} which is inside minimum sampling "
                    "radius {} (so will not include for correction)"
                    "".format(dist_to_defect, sampling_radius)
                )

        if len(for_correction) > 0:
            pot_alignment = np.mean(for_correction)
        else:
            logger.info(
                "No atoms sampled for_correction radius! Assigning potential alignment value of 0."
            )
            pot_alignment = 0.0

        self.metadata["potalign"] = pot_alignment
        pot_corr = -q * pot_alignment

        # log uncertainty stats:
        self.metadata["pot_corr_uncertainty_md"] = {
            "stats": stats.describe(for_correction)._asdict(),
            "number_sampled": len(for_correction),
        }
        self.metadata["pot_plot_data"] = pot_dict

        logger.info("Kumagai potential alignment (site averaging): %f", pot_alignment)
        logger.info("Kumagai potential alignment correction energy: %f eV", pot_corr)

        return pot_corr

    def get_real_summation(self, gamma, real_vectors):
        """
        Get real summation term from list of real-space vectors
        """
        real_part = 0
        invepsilon = np.linalg.inv(self.dielectric)
        rd_epsilon = np.sqrt(np.linalg.det(self.dielectric))

        for r_vec in real_vectors:
            if np.linalg.norm(r_vec) > 1e-8:
                loc_res = np.sqrt(np.dot(r_vec, np.dot(invepsilon, r_vec)))
                nmr = scipy.special.erfc(gamma * loc_res)  # pylint: disable=E1101
                real_part += nmr / loc_res

        real_part /= 4 * np.pi * rd_epsilon

        return real_part

    def get_recip_summation(self, gamma, recip_vectors, volume, r=[0.0, 0.0, 0.0]):
        """
        Get Reciprocal summation term from list of reciprocal-space vectors
        """
        recip_part = 0

        for g_vec in recip_vectors:
            # dont need to avoid G=0, because it will not be
            # in recip list (if generate_R_and_G_vecs is used)
            Gdotdiel = np.dot(g_vec, np.dot(self.dielectric, g_vec))
            summand = (
                np.exp(-Gdotdiel / (4 * (gamma**2)))
                * np.cos(np.dot(g_vec, r))
                / Gdotdiel
            )
            recip_part += summand

        recip_part /= volume

        return recip_part

    def plot(self, title=None, saved=False):
        """
        Plots the AtomicSite electrostatic potential against the Long range and short range models
        from Kumagai and Oba (doi: 10.1103/PhysRevB.89.195205)
        """
        if "pot_plot_data" not in self.metadata.keys():
            raise ValueError(
                "Cannot plot potential alignment before running correction!"
            )

        sampling_radius = self.metadata["sampling_radius"]
        site_dict = self.metadata["pot_plot_data"]
        potalign = self.metadata["potalign"]

        plt.figure()
        plt.clf()

        distances, sample_region = [], []
        Vqb_list, Vpc_list, diff_list = [], [], []
        for site_ind, site_dict in site_dict.items():
            dist = site_dict["dist_to_defect"]
            distances.append(dist)

            Vqb = site_dict["Vqb"]
            Vpc = site_dict["Vpc"]

            Vqb_list.append(Vqb)
            Vpc_list.append(Vpc)
            diff_list.append(Vqb - Vpc)

            if dist > sampling_radius:
                sample_region.append(Vqb - Vpc)

        plt.plot(
            distances,
            Vqb_list,
            color="r",
            marker="^",
            linestyle="None",
            label="$V_{q/b}$",
        )

        plt.plot(
            distances,
            Vpc_list,
            color="g",
            marker="o",
            linestyle="None",
            label="$V_{pc}$",
        )

        plt.plot(
            distances,
            diff_list,
            color="b",
            marker="x",
            linestyle="None",
            label="$V_{q/b}$ - $V_{pc}$",
        )

        x = np.arange(sampling_radius, max(distances) * 1.05, 0.01)
        y_max = max(max(Vqb_list), max(Vpc_list), max(diff_list)) + 0.1
        y_min = min(min(Vqb_list), min(Vpc_list), min(diff_list)) - 0.1
        plt.fill_between(
            x, y_min, y_max, facecolor="red", alpha=0.15, label="sampling region"
        )
        plt.axhline(y=potalign, linewidth=0.5, color="red", label="pot. align. / -q")

        plt.legend(loc=0)
        plt.axhline(y=0, linewidth=0.2, color="black")

        plt.ylim([y_min, y_max])
        plt.xlim([0, max(distances) * 1.1])

        plt.xlabel(r"Distance from defect ($\AA$)", fontsize=20)
        plt.ylabel("Potential (V)", fontsize=20)
        plt.title(str(title) + " atomic site potential plot", fontsize=20)

        if saved:
            plt.savefig(str(title) + "KumagaiESPavgPlot.pdf")
            return None
        return plt


def perform_es_corr(self, gamma, prec, lattice, charge):
    """
    Perform Electrostatic Kumagai Correction
    Args:
        gamma (float): Ewald parameter
        prec (int): Precision parameter for reciprical/real lattice vector generation
        lattice: Pymatgen Lattice object corresponding to defect supercell
        charge (int): Defect charge
    Return:
        Electrostatic Point Charge contribution to Kumagai Correction (float)
    """
    volume = lattice.volume

    g_vecs, recip_summation, r_vecs, real_summation = generate_R_and_G_vecs(
        gamma, [prec], lattice, self.dielectric
    )
    recip_summation = recip_summation[0]
    real_summation = real_summation[0]

    es_corr = (
        recip_summation
        + real_summation
        + self.get_potential_shift(gamma, volume)
        + self.get_self_interaction(gamma)
    )

    es_corr *= -(charge**2.0) * kumagai_to_V / 2.0  # [eV]
    return es_corr


def get_self_interaction(self, gamma):
    """
    Args:
        gamma ():

    Returns:
        Self-interaction energy of defect.
    """
    determ = np.linalg.det(self.dielectric)
    return -gamma / (2.0 * np.pi * np.sqrt(np.pi * determ))


def get_potential_shift(gamma, volume):
    """
    Args:
        gamma (float): Gamma
        volume (float): Volume.

    Returns:
        Potential shift for defect.
    """
    return -0.25 / (volume * gamma**2.0)
