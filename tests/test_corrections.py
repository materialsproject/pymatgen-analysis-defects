import pytest
from pymatgen.analysis.defects.corrections.freysoldt import (
    get_freysoldt_correction,
    plot_plnr_avg,
)


def test_freysoldt(data_Mg_Ga):
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


def test_freysoldt_sxdefect_compare(v_N_GaN):
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


# def test_kumagai(test_dir):
#     sb = get_structure_with_pot(test_dir / "Mg_Ga" / "bulk_sc")
#     sd0 = get_structure_with_pot(test_dir / "Mg_Ga" / "q=0")
#     sd1 = get_structure_with_pot(test_dir / "Mg_Ga" / "q=1")

#     res0 = get_efnv_correction(
#         0, sd0, sb, dielectric_tensor=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
#     )
#     assert res0.correction_energy == pytest.approx(0, abs=1e-4)

#     res1 = get_efnv_correction(
#         1, sd1, sb, dielectric_tensor=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
#     )
#     assert res1.correction_energy > 0
