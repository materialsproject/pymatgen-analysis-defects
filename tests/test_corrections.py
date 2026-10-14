import math

import numpy as np
import pytest
from pymatgen.analysis.defects.corrections.freysoldt import (
    get_freysoldt_correction,
    perform_es_corr,
    plot_plnr_avg,
)
from pymatgen.analysis.defects.corrections.kumagai import (
    get_efnv_correction,
    get_structure_with_pot,
)
from pymatgen.analysis.defects.utils import QModel
from pymatgen.core import Lattice


def test_freysoldt(data_Mg_Ga) -> None:
    """Older basic test for Freysoldt correction."""
    bulk_locpot = data_Mg_Ga["bulk_sc"]["locpot"]
    defect_locpot = data_Mg_Ga["q=0"]["locpot"]

    freysoldt_summary = get_freysoldt_correction(
        q=0,
        dielectric=14,
        defect_locpot=defect_locpot,
        bulk_locpot=bulk_locpot,
        defect_frac_coords=[0.5, 0.5, 0.5],
    )
    assert freysoldt_summary.correction_energy == pytest.approx(0, abs=1e-4)

    # simple check that the plotter works
    plot_plnr_avg(freysoldt_summary.metadata["plot_data"][0])

    # different ways to specify the locpot
    freysoldt_summary = get_freysoldt_correction(
        q=0,
        dielectric=14,
        lattice=defect_locpot.structure.lattice,
        defect_locpot=defect_locpot,
        bulk_locpot=bulk_locpot,
        defect_frac_coords=[0.5, 0.5, 0.5],
    )

    defect_locpot_dict = {str(k): defect_locpot.get_axis_grid(k) for k in [0, 1, 2]}
    bulk_locpot_dict = {str(k): bulk_locpot.get_axis_grid(k) for k in [0, 1, 2]}
    freysoldt_summary = get_freysoldt_correction(
        q=0,
        dielectric=14,
        lattice=defect_locpot.structure.lattice,
        defect_locpot=defect_locpot_dict,
        bulk_locpot=bulk_locpot_dict,
        defect_frac_coords=[0.5, 0.5, 0.5],
    )


def test_freysoldt_sxdefect_compare(v_N_GaN) -> None:
    """More detailed test for Freysoldt correction.

    Compare against results from the sxdefectalign tool from SPHInX.
    See the `freysoldt_correction.ipynb` notebook for details.
    """
    bulk_locpot = v_N_GaN["bulk_locpot"]
    defect_locpots = v_N_GaN["defect_locpots"]
    references = {
        -1: 0.366577,
        0: 0.0,
        1: 0.179924,
        2: 0.772761,
    }
    results = {
        q: get_freysoldt_correction(
            q=q,
            dielectric=5,
            bulk_locpot=bulk_locpot,
            defect_locpot=defect_locpots[q],
        ).correction_energy
        for q in range(-1, 3)
    }
    for q in range(-1, 3):
        assert results[q] == pytest.approx(references[q], abs=1e-3)


def test_freysoldt_anisotropic_dielectric() -> None:
    """Test anisotropic dielectric tensors in the Freysoldt correction.

    With the tensor treated as in ``sxdefectalign --tensor``, the point-charge energy
    matches the anisotropic point-charge Ewald energy (Kumagai & Oba, PRB 89, 195205
    (2014)) of 0.167 eV.
    """
    lattice = Lattice.cubic(10.0)
    es_corr = perform_es_corr(lattice, 1, [15, 15, 3], QModel())
    assert es_corr == pytest.approx(0.167, abs=5e-3)
    # isotropic tensors match scalar input
    assert perform_es_corr(lattice, 1, [10, 10, 10], QModel()) == perform_es_corr(
        lattice, 1, 10, QModel()
    )


def test_freysoldt_anisotropic_potential(v_N_GaN) -> None:
    """Test anisotropic dielectric tensors in the potential alignment.

    The planar-averaged model potential along each axis is screened by the dielectric
    constant along that axis, so it matches the isotropic model potential up to a
    constant (the G=0 term, screened by the spherically-averaged dielectric constant).
    """
    bulk_locpot = v_N_GaN["bulk_locpot"]
    defect_locpot = v_N_GaN["defect_locpots"][1]

    def get_correction(dielectric):
        return get_freysoldt_correction(
            q=1,
            dielectric=dielectric,
            bulk_locpot=bulk_locpot,
            defect_locpot=defect_locpot,
        )

    # isotropic tensors match scalar input
    scalar = get_correction(5).correction_energy
    assert get_correction([5, 5, 5]).correction_energy == pytest.approx(scalar)
    assert get_correction(5 * np.eye(3)).correction_energy == pytest.approx(scalar)

    # GaN is hexagonal, so axes 0 and 1 are in-plane and axis 2 is along c
    anisotropic = get_correction([5, 5, 10]).metadata["plot_data"]
    isotropic = {5: get_correction(5).metadata["plot_data"]}
    isotropic[10] = get_correction(10).metadata["plot_data"]
    for axis, eps in [(0, 5), (1, 5), (2, 10)]:
        diff = np.subtract(
            anisotropic[axis]["pot_plot_data"]["Vr"],
            isotropic[eps][axis]["pot_plot_data"]["Vr"],
        )
        assert np.ptp(diff) == pytest.approx(0, abs=1e-10)


def test_kumagai(test_dir):
    sb = get_structure_with_pot(test_dir / "Mg_Ga" / "bulk_sc")
    sd0 = get_structure_with_pot(test_dir / "Mg_Ga" / "q=0")
    sd1 = get_structure_with_pot(test_dir / "Mg_Ga" / "q=1")

    res0 = get_efnv_correction(
        0, sd0, sb, dielectric_tensor=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    assert res0.correction_energy == pytest.approx(0, abs=1e-4)

    # Bug-fix invariant (GH#219): pre-fix code returned a positive value
    # because defect/bulk site potentials were swapped at extraction. The
    # corrected wrapper must yield a finite, negative value for this q=+1
    # antisite fixture.
    res1 = get_efnv_correction(
        1, sd1, sb, dielectric_tensor=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    assert math.isfinite(res1.correction_energy)
    assert res1.correction_energy < 0

    # Snapshot value generated 2026-06-12 against the bundled Mg_Ga fixture.
    # Locks in pydefect numerics for regression detection only; not compared
    # against an external reference. Note the identity dielectric tensor —
    # chosen for fixture stability, not physical realism. Update if pydefect
    # numerics drift.
    assert res1.correction_energy == pytest.approx(-0.4898778, abs=1e-4)


def test_kumagai_vacancy(test_dir):
    """Regression test for GH#219: site potentials must not be swapped.

    With antisite defects (defect.num_sites == bulk.num_sites) the swap of
    `defect_structure` and `bulk_structure` on the `site.properties["potential"]`
    extraction is silent. With vacancies / interstitials the lengths differ and
    the bug surfaces as an IndexError downstream in pydefect.

    We synthesize a vacancy by removing a single site from the antisite q=0
    Mg_Ga test structure, so the defect has 31 sites and bulk has 32. The
    correction itself is meaningless for this synthetic input, but the call
    must not raise IndexError, and the potentials must be sourced from the
    correct structure.
    """
    sb = get_structure_with_pot(test_dir / "Mg_Ga" / "bulk_sc")
    sd0 = get_structure_with_pot(test_dir / "Mg_Ga" / "q=0")
    # Synthesize a vacancy: drop one site from the defect structure.
    sd_vac = sd0.copy()
    sd_vac.remove_sites([0])
    assert len(sd_vac) != len(sb), "test setup invalid: lengths must differ"

    # With the swap bug, building `defect_potentials` from `bulk_structure`
    # produces a 32-element array while `defect_structure` only has 31 sites,
    # which raises IndexError inside pydefect's make_efnv_correction. The
    # bug is fixed if this call returns without raising.
    get_efnv_correction(
        0, sd_vac, sb, dielectric_tensor=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )


def test_kumagai_missing():
    from pymatgen.analysis.defects.corrections import kumagai

    kumagai.__has_pydefect__ = False
    with pytest.raises(ImportError):
        kumagai._check_import_pydefect()
