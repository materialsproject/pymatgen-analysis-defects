"""Utilities for defects module."""
from __future__ import annotations

import bisect
import collections
import itertools
import logging
import math
import operator
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Generator

import numpy as np
from monty.json import MSONable
from numpy import typing as npt
from numpy.linalg import norm
from pymatgen.analysis.local_env import cn_opt_params
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element, get_el_sp
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import BandStructure, Procar, VolumetricData
from pymatgen.io.vasp.sets import get_valid_magmom_struct
from pymatgen.util.coord import pbc_diff
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import ConvexHull, Voronoi
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

_logger = logging.getLogger(__name__)
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
    """Find gzipped or non-gzipped versions of a file in a directory listing.

    Args:
        directory : A list of files in a directory.
        base_name : The base name of file file to find.
        allow_missing : Whether to error if no version of the file (gzipped or un-gzipped) can be found.

    Returns:
        Path | None: A path to the matched file. If ``allow_missing=True``
        and the file cannot be found, then ``None`` will be returned.
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
    """Group a list of unsortable objects.

    Args:
        list_in: A list of generic objects
        comp: (Default value = operator.eq) The comparator

    Returns:
        list[int]: list of labels for the input list

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
    """Get all local extrema fractional coordinates in charge density.

    Searching for local minimum by default. Note that sites are NOT grouped
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
    """Removed points that are too close to existing atoms in the structure.

    Args:
        fcoords (npt.ArrayLike): fractional coordinates of points to remove
        min_dist(float): The minimum distance that a vertex needs to be
            from existing atoms.

    Returns:
        fcoord (numpy.ndarray): The filtered coordinates.
    """
    s_fcoord = structure.frac_coords
    _logger.info(s_fcoord)
    dist_matrix = structure.lattice.get_all_distances(fcoords, s_fcoord)
    all_dist = np.min(dist_matrix, axis=1)
    return np.array(
        [fcoords[i] for i in range(len(fcoords)) if all_dist[i] >= min_dist]
    )


def cluster_nodes(
    fcoords: npt.ArrayLike, lattice: Lattice, tol: float = 0.2
) -> npt.NDArray:
    """Cluster nodes that are too close together using hiercharcal clustering.

    Args:
        fcoords (npt.ArrayLike): fractional coordinates of points to cluster.
        lattice (Lattice): The lattice of the structure.
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


class TopographyAnalyzer:
    """Topography Analyzer.

    This is a generalized module to perform topological analyses of a crystal
    structure using Voronoi tessellations. It can be used for finding potential
    interstitial sites. Applications including using these sites for
    inserting additional atoms or for analyzing diffusion pathways.

    Note that you typically want to do some preliminary postprocessing after
    the initial construction. The initial construction will create a lot of
    points, especially for determining potential insertion sites. Some helper
    methods are available to perform aggregation and elimination of nodes. A
    typical use is something like::

        a = TopographyAnalyzer(structure, ["O", "P"])
    """

    def __init__(
        self,
        structure,
        framework_ions,
        cations,
        image_tol=0.0001,
        max_cell_range=1,
        check_volume=True,
        constrained_c_frac=0.5,
        thickness=0.5,
        clustering_tol: float = 0.5,
        min_dist: float = 0.9,
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5,
    ):
        """Initialize the TopographyAnalyzer.

        Args:
            structure (Structure): An initial structure.
            framework_ions ([str]): A list of ions to be considered as a
                framework. Typically, this would be all anion species. E.g.,
                ["O", "S"].
            cations ([str]): A list of ions to be considered as non-migrating
                cations. E.g., if you are looking at Li3PS4 as a Li
                conductor, Li is a mobile species. Your cations should be [
                "P"]. The cations are used to exclude polyhedra from
                diffusion analysis since those polyhedra are already occupied.
            image_tol (float): A tolerance distance for the analysis, used to
                determine if something are actually periodic boundary images of
                each other. Default is usually fine.
            max_cell_range (int): This is the range of periodic images to
                construct the Voronoi tessellation. A value of 1 means that we
                include all points from (x +- 1, y +- 1, z+- 1) in the
                voronoi construction. This is because the Voronoi poly
                extends beyond the standard unit cell because of PBC.
                Typically, the default value of 1 works fine for most
                structures and is fast. But for really small unit
                cells with high symmetry, you may need to increase this to 2
                or higher.
            check_volume (bool): Set False when ValueError always happen after
                tuning tolerance.
            constrained_c_frac (float): Constraint the region where users want
                to do Topology analysis the default value is 0.5, which is the
                fractional coordinate of the cell
            thickness (float): Along with constrained_c_frac, limit the
                thickness of the regions where we want to explore. Default is
                0.5, which is mapping all the site of the unit cell.
            clustering_tol (float): Tolerance for clustering nodes. Default is
                0.5.
            min_dist (float): Minimum distance between nodes. Default is 0.9.
            ltol (float): Lattice parameter tolerance for structure matching.
                Default is 0.2.
            stol (float): Site tolerance for structure matching. Default is
                0.3.
            angle_tol (float): Angle tolerance for structure matching. Default
                is 5.

        """
        self.framework_ions = {get_el_sp(sp) for sp in framework_ions}
        self.cations = {get_el_sp(sp) for sp in cations}
        self.clustering_tol = clustering_tol
        self.min_dist = min_dist
        self.sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)

        # Let us first map all sites to the standard unit cell, i.e.,
        # 0 ≤ coordinates < 1.
        # structure = Structure.from_sites(structure, to_unit_cell=True)
        # lattice = structure.lattice

        # We could constrain the region where we want to dope/explore by setting
        # the value of constrained_c_frac and thickness. The default mode is
        # mapping all sites to the standard unit cell

        self.structure = structure.copy()
        # TODO: Structure is still being mutated something weird is going on but the code works.
        # remove oxidation state
        self.structure.remove_oxidation_states()

        constrained_sites = []
        for i, site in enumerate(self.structure):
            if (
                site.frac_coords[2] >= constrained_c_frac - thickness
                and site.frac_coords[2] <= constrained_c_frac + thickness
            ):
                constrained_sites.append(site)
        constrained_struct = Structure.from_sites(sites=constrained_sites)
        lattice = constrained_struct.lattice

        # Divide the sites into framework and non-framework sites.
        framework = []
        non_framework = []
        for site in constrained_struct:
            # site.species can be Composition SpecieLike
            if hasattr(site.species, "elements"):
                # Handle the case where site.species is a Composition
                els = [*map(Element, site.species.elements)]
            else:
                els = [*map(get_el_sp, site.species.keys())]
            if self.framework_ions.intersection(els):
                framework.append(site)
            else:
                non_framework.append(site)

        # We construct a supercell series of coords. This is because the
        # Voronoi polyhedra can extend beyond the standard unit cell. Using a
        # range of -2, -1, 0, 1 should be fine.
        coords = []
        cell_range = list(range(-max_cell_range, max_cell_range + 1))
        for shift in itertools.product(cell_range, cell_range, cell_range):
            for site in framework:
                shifted = site.frac_coords + shift
                coords.append(lattice.get_cartesian_coords(shifted))

        # Perform the voronoi tessellation.
        voro = Voronoi(coords)

        # Store a mapping of each voronoi node to a set of points.
        node_points_map = defaultdict(set)
        for pts, vs in voro.ridge_dict.items():
            for v in vs:
                node_points_map[v].update(pts)

        _logger.debug(f"{len(voro.vertices)} total Voronoi vertices")

        # Vnodes store all the valid voronoi polyhedra. Cation vnodes store
        # the voronoi polyhedra that are already occupied by existing cations.
        vnodes: list[VoronoiPolyhedron] = []
        cation_vnodes = []

        def get_mapping(poly):
            """Helper function.

            Checks if a vornoi poly is a periodic image of
            one of the existing voronoi polys.
            """
            for v in vnodes:
                if v.is_image(poly, image_tol):
                    return v
            return None

        # Filter all the voronoi polyhedra so that we only consider those
        # which are within the unit cell.
        for i, vertex in enumerate(voro.vertices):
            if i == 0:
                continue
            fcoord = lattice.get_fractional_coords(vertex)
            poly = VoronoiPolyhedron(lattice, fcoord, node_points_map[i], coords, i)
            if np.all([-image_tol <= c < 1 + image_tol for c in fcoord]):
                if len(vnodes) == 0:
                    vnodes.append(poly)
                else:
                    ref = get_mapping(poly)
                    if ref is None:
                        vnodes.append(poly)

        _logger.debug(f"{len(vnodes)} voronoi vertices in cell.")

        # Eliminate all voronoi nodes which are closest to existing cations.
        if len(cations) > 0:
            cation_coords = [
                site.frac_coords
                for site in non_framework
                if self.cations.intersection(site.species.keys())
            ]

            vertex_fcoords = [v.frac_coords for v in vnodes]
            dist_matrix = lattice.get_all_distances(cation_coords, vertex_fcoords)
            indices = np.where(dist_matrix == np.min(dist_matrix, axis=1)[:, None])[1]
            cation_vnodes = [v for i, v in enumerate(vnodes) if i in indices]
            vnodes = [v for i, v in enumerate(vnodes) if i not in indices]

        _logger.debug(f"{len(vnodes)} vertices in cell not with cation.")
        self.coords = coords
        self.vnodes = vnodes
        self.cation_vnodes = cation_vnodes
        self.framework = framework
        self.non_framework = non_framework
        if check_volume:
            self.check_volume()

    @property
    def labeled_sites(
        self,
    ) -> list[tuple[list[float], int]]:
        """Get a list of inserted structures and a list of structure matching labels.

        Returns:
            list[tuple[list[float], int]]: A list of tuples of the form (fcoords, label)
        """
        # Get a reasonablly reduced set of candidate sites first
        candidate_sites = [v.frac_coords for v in self.vnodes]
        return get_symmetry_labeled_structures(
            candidate_sites,
            host_structure=self.structure,
            working_ion="X",
            min_dist=self.min_dist,
            clustering_tol=self.clustering_tol,
            sm=self.sm,
        )

    def check_volume(self):
        """Basic check for volume of all voronoi poly sum to unit cell volume.

        Note that this does not apply after poly combination.
        """
        vol = sum(v.volume for v in self.vnodes) + sum(
            v.volume for v in self.cation_vnodes
        )
        if abs(vol - self.structure.volume) > 1e-8:  # pragma: no cover
            raise ValueError(
                "Sum of voronoi volumes is not equal to original volume of "
                "structure! This may lead to inaccurate results. You need to "
                "tweak the tolerance and max_cell_range until you get a "
                "correct mapping."
            )

    def get_structure_with_nodes(self):
        """Get the modified structure with the voronoi nodes inserted.

        The species is set as a DummySpecies X.
        """
        new_s = Structure.from_sites(self.structure)
        for v in self.vnodes:
            new_s.append("X", v.frac_coords)
        return new_s


class VoronoiPolyhedron:
    """Convenience container for a voronoi point in PBC and its associated polyhedron."""

    def __init__(self, lattice, frac_coords, polyhedron_indices, all_coords, name=None):
        """Initialize a VoronoiPolyhedron.

        Args:
            lattice (Lattice): Lattice of the structure.
            frac_coords (np.array): Fractional coordinates of the voronoi point.
            polyhedron_indices (list): Indices of the polyhedron vertices.
            all_coords (list): List of all polyhedron vertices.
            name (str): Name of the polyhedron.
        """
        self.lattice = lattice
        self.frac_coords = frac_coords
        self.polyhedron_indices = polyhedron_indices
        self.polyhedron_coords = np.array(all_coords)[list(polyhedron_indices), :]
        self.name = name

    def is_image(self, poly: VoronoiPolyhedron, tol: float) -> bool:
        """Check if poly is an image of the current polyhedron.

        Args:
            poly (VoronoiPolyhedron): Polyhedron to check.
            tol (float): Tolerance for image check.

        Returns:
            bool: True if poly is an image of the current polyhedron.
        """
        frac_diff = pbc_diff(poly.frac_coords, self.frac_coords)
        if not np.allclose(frac_diff, [0, 0, 0], atol=tol):
            return False
        to_frac = self.lattice.get_fractional_coords
        for c1 in self.polyhedron_coords:
            found = False
            for c2 in poly.polyhedron_coords:
                d = pbc_diff(to_frac(c1), to_frac(c2))
                if not np.allclose(d, [0, 0, 0], atol=tol):
                    found = True
                    break
            if not found:
                return False
        return True

    @property
    def coordination(self):
        """Coordination number."""
        return len(self.polyhedron_indices)

    @property
    def volume(self):
        """Volume of the polyhedron."""
        return calculate_vol(self.polyhedron_coords)

    def __str__(self):
        """String representation."""
        return f"Voronoi polyhedron {self.name}"


class ChargeInsertionAnalyzer(MSONable):
    """Object for insertions sites from charge density.

    Analyze the charge density and create new candidate structures by inserting at each charge minima
    The similar inserterd structures are given the same uniqueness label.

    .. note::
        The charge density analysis works best with AECCAR data since CHGCAR data
        often contains spurious local minima in the core. However you can still use CHGCAR
        with an appropriate ``max_avg_charge`` value.

        Since the user might want to rerun their analysis with different ``avg_charge`` and
        ``max_avg_charge`` values, we will generate and store all the ion-inserted structure
        and their uniqueness labels first and allow the user to get the filtered and labeled results.

        If you use this code please cite the following paper:
        J.-X. Shen et al.: npj Comput. Mater. 6, 1 (2020)
        https://www.nature.com/articles/s41524-020-00422-3

    Attributes:
        chgcar: The charge density object to analyze
        working_ion: The working ion to be inserted
        clustering_tol: Distance tolerance for grouping sites together
        ltol: StructureMatcher ltol parameter
        stol: StructureMatcher stol parameter
        angle_tol: StructureMatcher angle_tol parameter
        min_dist: Minimum distance between sites and the host atoms in Å.
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

        Returns:
            list[tuple[list[float], int]]: A list of tuples of the form (fcoords, label)
        """
        # Get a reasonablly reduced set of candidate sites first
        local_minima = get_local_extrema(self.chgcar, find_min=True)
        return get_symmetry_labeled_structures(
            sites=local_minima,
            host_structure=self.chgcar.structure,
            working_ion=self.working_ion,
            min_dist=self.min_dist,
            clustering_tol=self.clustering_tol,
            sm=self.sm,
        )

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


def _get_ipr(spin, k_index, procar):
    states = procar.data[spin][k_index, ...]
    flat_states = states.reshape(states.shape[0], -1)
    return 1 / np.sum(flat_states**2, axis=1)


def get_ipr_in_window(
    bandstructure: BandStructure,
    procar: Procar,
    band_window: int = 5,
) -> dict[Spin, npt.NDArray]:
    """Get the inverse participation ratio (IPR) of the states in a given band window.

    For a given number of bands above and below the Fermi level, the IPR is calculated.

    Args:
        bandstructure: The bandstructure object.
            The band just below the fermi level is used as the center of band window.
        procar: The procar object.
        k_index: The index of the k-point to use. If None, the IPR is averaged over all k-points.
        band_range: The number of bands above and blow the fermi level to include in the search window.

    Returns:
        dict[(int, int), npt.NDArray]: The IPR of the states in the band window keyed for each k-point and spin.
    """
    res = dict()
    for spin in bandstructure.bands.keys():
        s_index = 0 if spin == Spin.up else 1
        # last band that is fully below the fermi level
        last_occ_idx = bisect.bisect_left(
            bandstructure.bands[spin].max(1), bandstructure.efermi
        )
        lbound = max(last_occ_idx - band_window, 0)
        ubound = min(last_occ_idx + band_window, bandstructure.nb_bands)
        for k_idx, _ in enumerate(bandstructure.kpoints):
            ipr = _get_ipr(spin, k_idx, procar)
            res[(k_idx, s_index)] = np.stack(
                (np.arange(lbound, ubound), ipr[lbound:ubound])
            ).T
    return res


def get_localized_states(
    bandstructure: BandStructure,
    procar: Procar,
    band_window: int = 7,
) -> Generator[tuple[int, int, int, float], None, None]:
    """Find the (band, kpt, spin) index of the most localized state.

    Find the band index with the lowest inverse participation ratio (IPR) in a small window
    around the fermi level.  Since some of the core states can be very localized,
    we should only search near the band edges.  Report the result for each k-point and spin channel.

    Args:
        bandstructure: The bandstructure object.
            The band just below the fermi level is used as the center of band window.
        procar: The procar object.
        band_window: The number of bands above and blow the fermi level to include in the search window.

    Returns:
        tuple(int, int, int): Returns the (band, kpt, spin) index of the most localized state.
    """
    d_ib_ipr = get_ipr_in_window(bandstructure, procar, band_window)
    for (k_index, ispin), ib_ipr in d_ib_ipr.items():
        # find the index of the minimum IPR
        min_idx, val = min(ib_ipr, key=lambda x: x[1])
        yield (int(min_idx), k_index, ispin, val)


def sort_positive_definite(
    list_in: list, ref1: Any, ref2: Any, dist: Callable
) -> tuple[list, list[float]]:
    """Sort a list where we can only compute a positive-definite distance.

    Sometimes, we can only compute a positive-definite distance between two objects.
    (E.g., the displacement between two structures). In these cases, standard
    sorting algorithms will not work. Here, we accept two reference points to give
    some sense of direction. We then sort the list based on the distance between the
    reference points. Note: this only works if the list falls on a line of some sort.

    Args:
        list_in: The list to sort.
        ref1: The first reference point, this will be the `zero` point.
        ref2: The second reference point, this will determine the direction.
        dist: Some positive definite distance function.

    Returns:
        - the sorted list of objects.
        - the signed distance in the chosen direction.
    """
    d1 = [dist(el, ref1) for el in list_in]
    d2 = [dist(el, ref2) for el in list_in]
    D0 = dist(ref1, ref2)

    d_vs_s = []
    for q1, q2, el in zip(d1, d2, list_in):
        sign = +1
        if q1 < q2 and q2 > D0:
            sign = -1
        d_vs_s.append((sign * q1, el))
    d_vs_s.sort()
    distances, sorted_list = list(zip(*d_vs_s))
    return sorted_list, distances


def calculate_vol(coords: npt.NDArray):
    """Calculate volume given a set of points in 3D space.

    Args:
        coords: The coordinates of the points in 3D space.

    Returns:
        The volume of the convex hull of the points.
    """
    return ConvexHull(coords).volume


def get_symmetry_labeled_structures(
    sites: npt.NDArray,
    host_structure: Structure,
    working_ion: str,
    min_dist: float,
    clustering_tol: float,
    sm: StructureMatcher,
) -> list[tuple[list[float], int]]:
    """Get a list of inserted structures and a list of structure matching labels.

    The process is as follows:
    1. Read in the list of candidate sites.
    2. Remove any sites that collide with the host and cluster the remaining sites.
    2. Group the candidate sites by symmetry by inserting the working ion and structure matching.
    3. Label the groups by structure matching.

    Args:
        sites: The candidate sites to insert.
        host_structure: The host structure.
        working_ion: The working ion.
        min_dist: The minimum distance between the working ion and atoms in the host structure.
        clustering_tol: The tolerance for clustering the candidate sites.

    Returns:
        list[tuple[list[float], int]]: A list of tuples of the form (fcoords, label)
    """
    sites = remove_collisions(sites, structure=host_structure, min_dist=min_dist)
    sites = cluster_nodes(sites, lattice=host_structure.lattice, tol=clustering_tol)

    # Group the candidate sites by symmetry
    inserted_structs = []
    for fpos in sites:
        tmp_struct = host_structure.copy()
        get_valid_magmom_struct(tmp_struct, inplace=True, spin_mode="none")
        tmp_struct.insert(
            0,
            working_ion,
            fpos,
        )
        tmp_struct.sort()
        inserted_structs.append(tmp_struct)

    # Label the groups by structure matching
    site_labels = generic_groupby(inserted_structs, comp=sm.fit)
    return [*zip(sites.tolist(), site_labels)]


@dataclass
class CorrectionResult(MSONable):
    """A summary of the corrections applied to a structure.

    Attributes:
        electrostatic: The electrostatic correction.
        potential_alignment: The potential alignment correction.
        metadata: A dictionary of metadata for plotting and intermediate analysis.
    """

    correction_energy: float
    metadata: dict[Any, Any]
