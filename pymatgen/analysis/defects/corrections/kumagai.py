"""Kumagai defect correction module.

If this correction is used, please reference Kumagai and Oba's original paper
(doi: 10.1103/PhysRevB.89.195205) as well as Freysoldt's original
paper (doi: 10.1103/PhysRevLett.102.016402)

NOTE that equations 8 and 9 from Kumagai et al. reference are divided by (4 pi) to get SI units
"""

from __future__ import annotations

import logging
from collections import namedtuple

import numpy as np
import numpy.typing as npt
import scipy
from matplotlib import pyplot as plt
from pymatgen.core import Lattice, Site, Structure

from pymatgen.analysis.defects.utils import generate_R_and_G_vecs, kumagai_to_V

_logger = logging.getLogger(__name__)

# named tuple for storing result of kumagai correction
KumagaiSummary = namedtuple(
    "KumagaiSummary", ["electrostatic", "potential_alignment", "metadata"]
)

# class KumagaiCorrection(DefectCorrection):
#     """
#     A class for KumagaiCorrection class. Largely adapted from PyCDT code

#     If this correction is used, please reference Kumagai and Oba's original paper
#     (doi: 10.1103/PhysRevB.89.195205) as well as Freysoldt's original
#     paper (doi: 10.1103/PhysRevLett.102.016402)

#     NOTE that equations 8 and 9 from Kumagai et al. reference are divided by (4 pi) to get SI units
#     """

#     def __init__(self, dielectric_tensor, sampling_radius=None, gamma=None):
#         """
#         Initializes the Kumagai Correction
#         Args:
#             dielectric_tensor (float or 3x3 matrix): Dielectric constant for the structure

#             optional data that can be tuned:
#                 sampling_radius (float): radius (in Angstrom) which sites must be outside
#                     of to be included in the correction. Publication by Kumagai advises to
#                     use Wigner-Seitz radius of defect supercell, so this is default value.
#                 gamma (float): convergence parameter for gamma function.
#                     Code will automatically determine this if set to None.
#         """
#         self.metadata = {
#             "gamma": gamma,
#             "sampling_radius": sampling_radius,
#             "potalign": None,
#         }

#         if isinstance(dielectric_tensor, (int, float)):
#             self.dielectric = np.identity(3) * dielectric_tensor
#         else:
#             self.dielectric = np.array(dielectric_tensor)


def perform_es_corr(
    gamma: float, dielectric: npt.NDArray, prec: int, lattice: Lattice, charge: int
):
    """Perform Electrostatic Kumagai Correction.

    Args:
        gamma (float): Ewald parameter
        dielectric (npt.NDArray): Dielectric tensor
        prec (int): Precision parameter for reciprical/real lattice vector generation
        lattice: Pymatgen Lattice object corresponding to defect supercell
        charge (int): Defect charge

    Return:
        Electrostatic Point Charge contribution to Kumagai Correction (float)
    """
    volume = lattice.volume

    g_vecs, recip_summation, r_vecs, real_summation = generate_R_and_G_vecs(
        gamma, [prec], lattice, dielectric
    )
    recip_summation = recip_summation[0]
    real_summation = real_summation[0]

    es_corr = (
        recip_summation
        + real_summation
        + get_potential_shift(gamma, volume)
        + get_self_interaction(gamma, dielectric=dielectric)
    )

    es_corr *= -(charge**2.0) * kumagai_to_V / 2.0  # [eV]
    return es_corr


def get_self_interaction(gamma: float, dielectric: npt.NDArray) -> float:
    """Calculate self interaction.

    Args:
        gamma (float): Modified Ewald parameter.
        dielectric (np.ndarray): Dielectric tensor.

    Returns:
        float: Self-interaction energy of defect.
    """
    determ = np.linalg.det(dielectric)
    return -gamma / (2.0 * np.pi * np.sqrt(np.pi * determ))


def get_potential_shift(gamma: float, volume: float) -> float:
    """Calculate potential shift.

    Args:
        gamma (float): Gamma
        volume (float): Volume.

    Returns:
        Potential shift for defect.
    """
    return -0.25 / (volume * gamma**2.0)


def get_real_summation(gamma: float, dielectric: npt.NDArray, real_vectors) -> float:
    """Get real summation term from list of real-space vectors.

    Args:
        gamma (float): Ewald parameter
        dielectric (npt.NDArray): Dielectric tensor
        real_vectors (list): List of real-space vectors

    Returns:
        float: Real summation term
    """
    real_part = 0
    invepsilon = np.linalg.inv(dielectric)
    rd_epsilon = np.sqrt(np.linalg.det(dielectric))

    for r_vec in real_vectors:
        if np.linalg.norm(r_vec) > 1e-8:
            loc_res = np.sqrt(np.dot(r_vec, np.dot(invepsilon, r_vec)))
            nmr = scipy.special.erfc(gamma * loc_res)  # pylint: disable=E1101
            real_part += nmr / loc_res

    real_part /= 4 * np.pi * rd_epsilon

    return real_part


def get_recip_summation(
    gamma, dielectric, recip_vectors, volume, r: list | None = None
) -> float:
    """Get Reciprocal summation term from list of reciprocal-space vectors.

    Args:
        gamma (float): Ewald parameter
        dielectric (np.ndarray): Dielectric tensor
        recip_vectors (list): List of reciprocal-space vectors
        volume (float): Volume of supercell
        r (list): List of reciprocal-space vectors default to [0,0,0]
    """
    r = r if r is not None else [0.0, 0.0, 0.0]
    recip_part = 0

    for g_vec in recip_vectors:
        # dont need to avoid G=0, because it will not be
        # in recip list (if generate_R_and_G_vecs is used)
        Gdotdiel = np.dot(g_vec, np.dot(dielectric, g_vec))
        summand = (
            np.exp(-Gdotdiel / (4 * (gamma**2))) * np.cos(np.dot(g_vec, r)) / Gdotdiel
        )
        recip_part += summand

    recip_part /= volume

    return recip_part


def tune_for_gamma(lattice, epsilon):
    """Get gamma value for Kumagai correction.

    This tunes the gamma parameter for Kumagai anisotropic
    Ewald calculation. Method is to find a gamma parameter which generates a similar
    number of reciprocal and real lattice vectors,
    given the suggested cut off radii by Kumagai and Oba

    Args:
        lattice (Lattice): pymatgen lattice object
        epsilon (np.ndarray): dielectric tensor
    """
    _logger.debug("Converging for ewald parameter...")
    prec = 25  # a reasonable precision to tune gamma for

    gamma = (2 * np.average(lattice.abc)) ** (-1 / 2.0)
    recip_set, _, real_set, _ = generate_R_and_G_vecs(gamma, prec, lattice, epsilon)
    recip_set = recip_set[0]
    real_set = real_set[0]

    _logger.debug(
        "First approach with gamma ={}\nProduced {} real vecs and {} recip "
        "vecs.".format(gamma, len(real_set), len(recip_set))
    )

    while (
        float(len(real_set)) / len(recip_set) > 1.05
        or float(len(recip_set)) / len(real_set) > 1.05
    ):
        gamma *= (float(len(real_set)) / float(len(recip_set))) ** 0.17
        _logger.debug("\tNot converged...Try modifying gamma to {}.".format(gamma))
        recip_set, _, real_set, _ = generate_R_and_G_vecs(gamma, prec, lattice, epsilon)
        recip_set = recip_set[0]
        real_set = real_set[0]
        _logger.debug(
            "Now have {} real vecs and {} recip vecs.".format(
                len(real_set), len(recip_set)
            )
        )

    _logger.debug("Converged with gamma = {}".format(gamma))

    return gamma


def get_kumagai_correction(
    gamma: float,
    dielectric: npt.NDArray,
    q: int,
    bulk_atomic_site_averages: list,
    defect_atomic_site_averages: list,
    site_matching_indices: list,
    defect_sc_structure: Structure,
    defect_frac_sc_coords: npt.ArrayLike,
    sampling_radius: float | None = None,
):
    """Gets the Kumagai correction for a defect entry.

    Args:
        gamma (float): Ewald parameter
        dielectric (npt.NDArray): dielectric tensor
        sampling_radius (float): sampling radius
        q (int): charge of defect
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
    # bulk_atomic_site_averages = entry.parameters["bulk_atomic_site_averages"]
    # defect_atomic_site_averages = entry.parameters["defect_atomic_site_averages"]
    # site_matching_indices = entry.parameters["site_matching_indices"]
    # defect_sc_structure = entry.parameters["initial_defect_structure"]
    # defect_frac_sc_coords = entry.parameters["defect_frac_sc_coords"]

    lattice = defect_sc_structure.lattice
    volume = lattice.volume

    if not gamma:
        gamma = tune_for_gamma(lattice, dielectric)

    prec_set = [25, 28]
    g_vecs, recip_summation, r_vecs, real_summation = generate_R_and_G_vecs(
        gamma, prec_set, lattice, dielectric
    )

    pot_shift = get_potential_shift(gamma, volume)
    si = get_self_interaction(gamma, dielectric)
    es_corr = [
        (real_summation[ind] + recip_summation[ind] + pot_shift + si)
        for ind in range(2)
    ]

    # increase precision if correction is not converged yet
    # TODO: allow for larger prec_set to be tried if this fails
    if abs(es_corr[0] - es_corr[1]) > 0.0001:
        _logger.debug(
            "Es_corr summation not converged! ({} vs. {})\nTrying a larger prec_set...".format(
                es_corr[0], es_corr[1]
            )
        )
        prec_set = [30, 35]
        g_vecs, recip_summation, r_vecs, real_summation = generate_R_and_G_vecs(
            gamma, prec_set, lattice, dielectric
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
    if sampling_radius is not None:
        wz = lattice.get_wigner_seitz_cell()
        dist = []
        for facet in wz:
            midpt = np.mean(np.array(facet), axis=0)
            dist.append(np.linalg.norm(midpt))
        sampling_radius = min(dist)

    # assemble site_list based on matching indices
    # [[defect_site object, Vqb for site], .. repeat for all non defective sites]
    site_list = []
    for bs_ind, ds_ind in site_matching_indices:
        Vqb = -(
            defect_atomic_site_averages[int(ds_ind)]
            - bulk_atomic_site_averages[int(bs_ind)]
        )
        site_list.append((defect_sc_structure[int(ds_ind)], Vqb))

    pot_corr, metadata = perform_pot_corr(
        defect_sc_structure,
        defect_frac_sc_coords,
        site_list,
        sampling_radius,
        q,
        r_vecs[0],
        g_vecs[0],
        gamma,
        dielectric,
    )

    return KumagaiSummary(
        electrostatic=es_corr,
        potential_alignment=pot_corr / (-q) if q else 0.0,
        metadata=metadata,
    )


def perform_pot_corr(
    defect_structure: Structure,
    defect_frac_coords: npt.ArrayLike,
    site_list: list[tuple[Site, int]],
    sampling_radius: float,
    q: int,
    r_vecs: list,
    g_vecs: list,
    gamma: float,
    dielectric: npt.NDArray,
):
    """Function performing potential alignment in manner described by Kumagai et al.

    Args:
        defect_structure: Pymatgen Structure object corresponding to the defect supercell
        defect_frac_coords (array): Defect Position in fractional coordinates of the supercell given in bulk_structure
        site_list: List of corresponding site index values for
            bulk and defect site structures EXCLUDING the defect site itself
            (ex. [[bulk structure site index, defect structure"s corresponding site index], ... ]
        sampling_radius (float): Radius (in Angstrom) which sites must be outside
            of to be included in the correction. Publication by Kumagai advises to
            use Wigner-Seitz radius of defect supercell, so this is default value.
        q (int): Defect charge
        r_vecs: List of real lattice vectors to use in summation
        g_vecs: List of reciprocal lattice vectors to use in summation
        gamma (float): Modified Ewald parameter from Kumagai et al. 2014
        dielectric (np.ndarray): Dielectric tensor

    Return:
        float:
            Potential alignment contribution to Kumagai Correction (float)
        dict:
            metadata for plotting and analysis.
    """
    volume = defect_structure.lattice.volume
    potential_shift = get_potential_shift(gamma, volume)

    pot_dict = {}  # keys will be site index in the defect structure
    for_correction = []  # region to sample for correction
    metadata = {}  # metadata for plotting and analysis

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

        real_sum = get_real_summation(gamma, dielectric, relative_real_vectors)
        recip_sum = get_recip_summation(
            gamma, dielectric, g_vecs, volume, r=vec_defect_to_site[:]
        )

        Vpc = (real_sum + recip_sum + potential_shift) * kumagai_to_V * q

        defect_struct_index = defect_structure.index(site)
        pot_dict[defect_struct_index] = {
            "Vpc": Vpc,
            "Vqb": Vqb,
            "dist_to_defect": dist_to_defect,
        }

        _logger.debug(
            f"For atom {defect_struct_index}\n\tbulk/defect DFT potential difference = {Vqb}"
        )
        _logger.debug(f"\tanisotropic model charge: {Vpc}")
        _logger.debug(f"\t\treciprocal part: {recip_sum * kumagai_to_V * q}")
        _logger.debug(f"\t\treal part: {real_sum * kumagai_to_V * q}")
        _logger.debug(
            f"\t\tself interaction part: {potential_shift * kumagai_to_V * q}"
        )
        _logger.debug(f"\trelative_vector to defect: {vec_defect_to_site}")

        if dist_to_defect > sampling_radius:
            _logger.debug(
                "\tdistance to defect is {} which is outside minimum sampling "
                "radius {}".format(dist_to_defect, sampling_radius)
            )
            for_correction.append(Vqb - Vpc)
        else:
            _logger.debug(
                "\tdistance to defect is {} which is inside minimum sampling "
                "radius {} (so will not include for correction)"
                "".format(dist_to_defect, sampling_radius)
            )

    if len(for_correction) > 0:
        pot_alignment = np.mean(for_correction)
    else:
        _logger.info(
            "No atoms sampled for_correction radius! Assigning potential alignment value of 0."
        )
        pot_alignment = 0.0

    metadata["potalign"] = pot_alignment
    pot_corr = -q * pot_alignment

    # log uncertainty stats:
    metadata["pot_corr_uncertainty_md"] = {
        "stats": scipy.stats.describe(for_correction)._asdict(),
        "number_sampled": len(for_correction),
    }
    metadata["pot_plot_data"] = pot_dict

    _logger.info("Kumagai potential alignment (site averaging): %f", pot_alignment)
    _logger.info("Kumagai potential alignment correction energy: %f eV", pot_corr)

    return pot_corr, metadata


def plot(ks: KumagaiSummary, title=None, saved=False):
    """Summary plotter for Kumagai Correction.

    Plots the AtomicSite electrostatic potential against the Long range and short range models
    from Kumagai and Oba (doi: 10.1103/PhysRevB.89.195205)

    Args:
        ks (KumagaiSummary): KumagaiSummary object
        title (str): Title for plot
        saved (bool): Whether to save the plot to a file

    Returns:
        matplotlib.pyplot: Plotting module state.
    """
    if "pot_plot_data" not in ks.metadata.keys():
        raise ValueError("Cannot plot potential alignment before running correction!")

    sampling_radius = ks.metadata["sampling_radius"]
    site_dict = ks.metadata["pot_plot_data"]
    potalign = ks.metadata["potalign"]

    plt.figure()
    plt.clf()

    distances, sample_region = [], []
    Vqb_list, Vpc_list, diff_list = [], [], []
    for _site_ind, sd in site_dict.items():
        dist = sd["dist_to_defect"]
        distances.append(dist)

        Vqb = sd["Vqb"]
        Vpc = sd["Vpc"]

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
