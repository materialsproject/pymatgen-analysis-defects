"""Utilities for defects module."""
from __future__ import annotations

import collections
import logging
import math
import operator
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Generator

import numpy as np
from monty.json import MSONable
from numpy import typing as npt
from numpy.linalg import norm
from pymatgen.analysis.local_env import cn_opt_params
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import VolumetricData
from pymatgen.io.vasp.sets import get_valid_magmom_struct
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

try:
    from skimage.feature import peak_local_max

    peak_local_max_found = True
except ImportError:
    peak_local_max_found = False

__author__ = (
    "Jimmy-Xuan Shen, Danny Broberg, Shyam Dwaraknath, Bharat Medasani, Nils Zimmermann, "
    "Geoffroy Hautier"
)
__maintainer__ = "Jimmy-Xuan Shen"
__email__ = "jmmshn@gmail.com"
__status__ = "Development"
__date__ = "Aug 15, 2022"

logger = logging.getLogger(__name__)
hart_to_ev = 27.2114
ang_to_bohr = 1.8897
invang_to_ev = 3.80986
kumagai_to_V = 1.809512739e2  # = Electron charge * 1e10 / VacuumPermittivity Constant

motif_cn_op = {}
for cn, di in cn_opt_params.items():
    for motif, li in di.items():
        motif_cn_op[motif] = {"cn": int(cn), "optype": li[0]}
        motif_cn_op[motif]["params"] = deepcopy(li[1]) if len(li) > 1 else None


class QModel(MSONable):
    """Model for the defect charge distribution.

    A combination of exponential tail and gaussian distribution is used
    (see Freysoldt (2011), DOI: 10.1002/pssb.201046289 )

    q_model(r) = q [x exp(-r/gamma) + (1-x) exp(-r^2/beta^2)]
    without normalization constants

    By default, gaussian distribution with 1 Bohr width is assumed.
    If defect charge is more delocalized, exponential tail is suggested.
    """

    def __init__(self, beta=1.0, expnorm=0.0, gamma=1.0):
        """Initialize the model.

        Args:
            beta: Gaussian decay constant. Default value is 1 Bohr.
                When delocalized (eg. diamond), 2 Bohr is more appropriate.
            expnorm: Weight for the exponential tail in the range of [0-1].
                Default is 0.0 indicating no tail. For delocalized charges ideal value
                is around 0.54-0.6.
            gamma: Exponential decay constant
        """
        self.beta = beta
        self.expnorm = expnorm
        self.gamma = gamma

        self.beta2 = beta * beta
        self.gamma2 = gamma * gamma
        if expnorm and not gamma:
            raise ValueError("Please supply exponential decay constant.")

    def rho_rec(self, g2):
        """Reciprocal space model charge value.

        Reciprocal space model charge value, for input squared reciprocal vector.

        Args:
            g2: Square of reciprocal vector

        Returns:
            Charge density at the reciprocal vector magnitude
        """
        return self.expnorm / np.sqrt(1 + self.gamma2 * g2) + (
            1 - self.expnorm
        ) * np.exp(-0.25 * self.beta2 * g2)

    @property
    def rho_rec_limit0(self):
        """Reciprocal space model charge value.

        Close to reciprocal vector 0 .
        rho_rec(g->0) -> 1 + rho_rec_limit0 * g^2
        """
        return -2 * self.gamma2 * self.expnorm - 0.25 * self.beta2 * (1 - self.expnorm)


def eV_to_k(energy):
    """Convert energy to reciprocal vector magnitude k via hbar*k^2/2m.

    Args:
        a: Energy in eV.

    Returns:
        (double) Reciprocal vector magnitude (units of 1/Bohr).
    """
    return math.sqrt(energy / invang_to_ev) * ang_to_bohr


def genrecip(a1, a2, a3, encut) -> Generator[npt.ArrayLike, None, None]:
    """Generate reciprocal lattice vectors within the energy cutoff.

    Args:
        a1: Lattice vector a (in Bohrs)
        a2: Lattice vector b (in Bohrs)
        a3: Lattice vector c (in Bohrs)
        encut: energy cut off in eV

    Returns:
        reciprocal lattice vectors with energy less than encut
    """
    vol = np.dot(a1, np.cross(a2, a3))  # 1/bohr^3
    b1 = (2 * np.pi / vol) * np.cross(a2, a3)  # units 1/bohr
    b2 = (2 * np.pi / vol) * np.cross(a3, a1)
    b3 = (2 * np.pi / vol) * np.cross(a1, a2)

    # create list of recip space vectors that satisfy |i*b1+j*b2+k*b3|<=encut
    G_cut = eV_to_k(encut)
    # Figure out max in all recipricol lattice directions
    i_max = int(math.ceil(G_cut / norm(b1)))
    j_max = int(math.ceil(G_cut / norm(b2)))
    k_max = int(math.ceil(G_cut / norm(b3)))

    # Build index list
    i = np.arange(-i_max, i_max)
    j = np.arange(-j_max, j_max)
    k = np.arange(-k_max, k_max)

    # Convert index to vectors using meshgrid
    indices = np.array(np.meshgrid(i, j, k)).T.reshape(-1, 3)
    # Multiply integer vectors to get recipricol space vectors
    vecs = np.dot(indices, [b1, b2, b3])
    # Calculate radii of all vectors
    radii = np.sqrt(np.einsum("ij,ij->i", vecs, vecs))

    # Yield based on radii
    for vec, r in zip(vecs, radii):
        if r < G_cut and r != 0:
            yield vec


def generate_reciprocal_vectors_squared(a1, a2, a3, encut):
    """Generate Reciprocal vectors squared.

    Generate reciprocal vector magnitudes within the cutoff along the specified
    lattice vectors.

    Args:
        a1: Lattice vector a (in Bohrs)
        a2: Lattice vector b (in Bohrs)
        a3: Lattice vector c (in Bohrs)
        encut: Reciprocal vector energy cutoff

    Returns:
        [[g1^2], [g2^2], ...] Square of reciprocal vectors (1/Bohr)^2
        determined by a1, a2, a3 and whose magntidue is less than gcut^2.
    """
    for vec in genrecip(a1, a2, a3, encut):
        yield np.dot(vec, vec)


def converge(f, step, tol, max_h):
    """Simple newton iteration based convergence function."""
    g = f(0)
    dx = 10000
    h = step
    while dx > tol:
        g2 = f(h)
        dx = abs(g - g2)
        g = g2
        h += step

        if h > max_h:
            raise Exception(f"Did not converge before {h}")
    return g


def get_zfile(
    directory: Path, base_name: str, allow_missing: bool = False
) -> Path | None:
    """
    Find gzipped or non-gzipped versions of a file in a directory listing.

    Parameters
    ----------
    directory : list of Path
        A list of files in a directory.
    base_name : str
        The base name of file file to find.
    allow_missing : bool
        Whether to error if no version of the file (gzipped or un-gzipped) can be found.
    Returns
    -------
    Path or None
        A path to the matched file. If ``allow_missing=True`` and the file cannot be
        found, then ``None`` will be returned.
    """
    for file in directory.glob(f"{base_name}*"):
        if base_name == file.name:
            return file
        elif base_name + ".gz" == file.name:
            return file
        elif base_name + ".GZ" == file.name:
            return file

    if allow_missing:
        return None

    raise FileNotFoundError(f"Could not find {base_name} or {base_name}.gz file.")


def generic_groupby(list_in, comp=operator.eq):
    """
    Group a list of unsortable objects
    Args:
        list_in: A list of generic objects
        comp: (Default value = operator.eq) The comparator
    Returns:
        [int] list of labels for the input list
    """
    list_out = [None] * len(list_in)
    label_num = 0
    for i1, ls1 in enumerate(list_out):
        if ls1 is not None:
            continue
        list_out[i1] = label_num
        for i2, ls2 in list(enumerate(list_out))[(i1 + 1) :]:
            if comp(list_in[i1], list_in[i2]):
                if ls2 is None:
                    list_out[i2] = list_out[i1]
                else:
                    list_out[i1] = ls2
                    label_num -= 1
        label_num += 1
    return list_out


def get_local_extrema(chgcar: VolumetricData, find_min: bool = True) -> npt.NDArray:
    """
    Get all local extrema fractional coordinates in charge density,
    searching for local minimum by default. Note that sites are NOT grouped
    symmetrically.

    Args:
        find_min (bool): True to find local minimum else maximum, otherwise
            find local maximum.

    Returns:
        extrema_coords (list): list of fractional coordinates corresponding
            to local extrema.
    """

    if find_min:
        sign = -1
    else:
        sign = 1

    # Make 3x3x3 supercell
    # This is a trick to resolve the periodical boundary issue.
    # TODO: Add code to pyrho for max and min filtering.
    total_chg = sign * chgcar.data["total"]
    total_chg = np.tile(total_chg, reps=(3, 3, 3))
    coordinates = peak_local_max(total_chg, min_distance=1)

    # Remove duplicated sites introduced by supercell.
    f_coords = [coord / total_chg.shape * 3 for coord in coordinates]
    f_coords = [
        f - 1 for f in f_coords if all(np.array(f) < 2) and all(np.array(f) >= 1)
    ]

    return np.array(f_coords)


def remove_collisions(
    fcoords: npt.NDArray, structure: Structure, min_dist: float = 0.9
) -> npt.NDArray:
    """
    Removed points that are too close to existing atoms in the structure

    Args:
        fcoords (npt.ArrayLike): fractional coordinates of points to remove
        min_dist(float): The minimum distance that a vertex needs to be
            from existing atoms.

    Returns:
        fcoord (numpy.ndarray): The filtered coordinates.
    """
    s_fcoord = structure.frac_coords
    logger.info(s_fcoord)
    dist_matrix = structure.lattice.get_all_distances(fcoords, s_fcoord)
    all_dist = np.min(dist_matrix, axis=1)
    return np.array(
        [fcoords[i] for i in range(len(fcoords)) if all_dist[i] >= min_dist]
    )


def cluster_nodes(
    fcoords: npt.ArrayLike, lattice: Lattice, tol: float = 0.2
) -> npt.NDArray:
    """
    Cluster nodes that are too close together using a tol.

    Args:
        tol (float): A distance tolerance. PBC is taken into account.
    """
    # Manually generate the distance matrix (which needs to take into
    # account PBC.
    dist_matrix = np.array(lattice.get_all_distances(fcoords, fcoords))
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    for i in range(len(dist_matrix)):
        dist_matrix[i, i] = 0
    condensed_m = squareform(dist_matrix)
    z = linkage(condensed_m)
    cn = fcluster(z, tol, criterion="distance")
    merged_fcoords = []

    for n in set(cn):
        frac_coords = []
        for i, j in enumerate(np.where(cn == n)[0]):
            if i == 0:
                frac_coords.append(fcoords[j])
            else:
                fcoord = fcoords[j]
                # We need the image to combine the frac_coords properly.
                d, image = lattice.get_distance_and_image(frac_coords[0], fcoord)
                frac_coords.append(fcoord + image)
        merged_fcoords.append(np.average(frac_coords, axis=0))

    merged_fcoords = [f - np.floor(f) for f in merged_fcoords]
    merged_fcoords = [f * (np.abs(f - 1) > 1e-15) for f in merged_fcoords]
    # the second line for fringe cases like
    # np.array([ 5.0000000e-01 -4.4408921e-17  5.0000000e-01])
    # where the shift to [0,1) does not work due to float precision

    return np.array(merged_fcoords)


def get_avg_chg(
    chgcar: VolumetricData, fcoord: npt.ArrayLike, radius: float = 0.4
) -> float:
    """Get the average charge in a sphere.

    Args:
        chgcar: The charge density.
        fcoord: The fractional coordinates of the center of the sphere.
        radius: The radius of the sphere in Angstroms.

    Returns:
        The average charge in the sphere.


    """
    # makesure fcoord is an array
    fcoord = np.array(fcoord)

    def _dist_mat(pos_frac):
        # return a matrix that contains the distances
        aa = np.linspace(0, 1, len(chgcar.get_axis_grid(0)), endpoint=False)
        bb = np.linspace(0, 1, len(chgcar.get_axis_grid(1)), endpoint=False)
        cc = np.linspace(0, 1, len(chgcar.get_axis_grid(2)), endpoint=False)
        AA, BB, CC = np.meshgrid(aa, bb, cc, indexing="ij")
        dist_from_pos = chgcar.structure.lattice.get_all_distances(
            fcoords1=np.vstack([AA.flatten(), BB.flatten(), CC.flatten()]).T,
            fcoords2=pos_frac,
        )
        return dist_from_pos.reshape(AA.shape)

    if np.any(fcoord < 0) or np.any(fcoord > 1):
        raise ValueError("f_coords must be in [0,1)")
    mask = _dist_mat(fcoord) < radius
    vol_sphere = chgcar.structure.volume * (mask.sum() / chgcar.ngridpts)
    avg_chg = np.sum(chgcar.data["total"] * mask) / mask.size / vol_sphere
    return avg_chg


class ChargeInsertionAnalyzer(MSONable):
    """
    Analyze the charge density and create new candidate structures by inserting at each charge minima
    The similar inserterd structures are given the same uniqueness label.

    .. note::
        The charge density analysis works best with AECCAR data since CHGCAR data
        often contains spurious local minima in the core. However you can still use CHGCAR
        with an appropriate ``max_avg_charge`` value.

        Since the user might want to rerun their analysis with different ``avg_charge`` and ``max_avg_charge`` values,
        we will generate and store all the ion-inserted structure and their uniqueness labels first and allow
        the user to get the filtered and labeled results.

        If you use this code please cite the following paper:
        J.-X. Shen et al.: npj Comput. Mater. 6, 1 (2020)
        https://www.nature.com/articles/s41524-020-00422-3
    """

    def __init__(
        self,
        chgcar: VolumetricData,
        working_ion: str = "Li",
        clustering_tol: float = 0.5,
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5,
        min_dist: float = 0.9,
    ):
        """
        Args:
            chgcar: The charge density object to analyze
            working_ion: The working ion to be inserted
            clustering_tol: Distance tolerance for grouping sites together
            ltol: StructureMatcher ltol parameter
            stol: StructureMatcher stol parameter
            angle_tol: StructureMatcher angle_tol parameter
            min_dist: Minimum distance between sites and the host atoms in Å.
        """
        self.chgcar = chgcar
        self.working_ion = working_ion
        self.sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
        self.clustering_tol = clustering_tol
        self.min_dist = min_dist

    @cached_property
    def labeled_sites(
        self,
    ) -> list[tuple[list[float], int]]:
        """Get a list of inserted structures and a list of structure matching labels.

        The process is as follows:
        1. Get a list of candidate sites by finiding the local minima
        2. Group the candidate sites by symmetry
        3. Label the groups by structure matching
        4. Since the average charge density is the most expensive part,
        we will leave this until the end.

        Returns:
            list[tuple[list[float], int]]: A list of tuples of the form (fcoords, label)
        """
        # Get a reasonablly reduced set of candidate sites first
        local_minima = get_local_extrema(self.chgcar, find_min=True)
        local_minima = remove_collisions(
            local_minima, structure=self.chgcar.structure, min_dist=self.min_dist
        )
        local_minima = cluster_nodes(
            local_minima, lattice=self.chgcar.structure.lattice, tol=self.clustering_tol
        )

        # Group the candidate sites by symmetry
        inserted_structs = []
        for fpos in local_minima:
            tmp_struct = self.chgcar.structure.copy()
            get_valid_magmom_struct(tmp_struct, inplace=True, spin_mode="none")
            tmp_struct.insert(
                0,
                self.working_ion,
                fpos,
            )
            tmp_struct.sort()
            inserted_structs.append(tmp_struct)

        # Label the groups by structure matching
        site_labels = generic_groupby(inserted_structs, comp=self.sm.fit)
        return [*zip(local_minima.tolist(), site_labels)]

    @cached_property
    def local_minima(self) -> list[npt.ArrayLike]:
        """Get the full list of local minima."""
        return [s for s, l in self.labeled_sites]

    def filter_and_group(
        self, avg_radius: float = 0.4, max_avg_charge: float = 1.0
    ) -> list[tuple[float, list[list[float]]]]:
        """Filter and group the insertion sites by average charge.

        Args:
            avg_radius: The radius used to calculate average charge density.
            max_avg_charge: Do no consider local minmas with avg charge above this value.

        Returns:
            list[tuple[float, list[int]]]: The list of ``(avg_charge, site_group)`` tuples
            where ``site_group`` are the positions of the local minima.
        """
        # measure the charge density at one representative site of each group
        lab_groups = collections.defaultdict(list)
        for idx, (_, lab) in enumerate(self.labeled_sites):
            lab_groups[lab].append(idx)

        avg_chg_first_member = {}
        for lab, g in lab_groups.items():
            avg_chg_first_member[lab] = get_avg_chg(
                self.chgcar, fcoord=self.local_minima[g[0]], radius=avg_radius
            )

        res = []
        for lab, avg_chg in sorted(avg_chg_first_member.items(), key=lambda x: x[1]):
            if avg_chg > max_avg_charge:
                break
            res.append((avg_chg, [self.local_minima[idx] for idx in lab_groups[lab]]))

        return res
