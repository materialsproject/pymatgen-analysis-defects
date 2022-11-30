import pytest

from pymatgen.analysis.defects.corrections.freysoldt import (
    get_freysoldt_correction,
    plot_plnr_avg,
)


def test_freysoldt(data_Mg_Ga):
    bulk_locpot = data_Mg_Ga["bulk_sc"]["locpot"]
    defect_locpot = data_Mg_Ga[f"q=0"]["locpot"]

    freysoldt_summary = get_freysoldt_correction(
        q=0,
        dielectric=14,
        defect_locpot=defect_locpot,
        bulk_locpot=bulk_locpot,
        defect_frac_coords=[0.5, 0.5, 0.5],
    )
    assert freysoldt_summary.electrostatic == pytest.approx(0, abs=1e-4)
    assert freysoldt_summary.potential_alignment == pytest.approx(0, abs=1e-4)

    # simple check that the plotter works
    plot_plnr_avg(freysoldt_summary.metadata[0])

    # different ways to specify the locpot
    freysoldt_summary = get_freysoldt_correction(
        q=0,
        dielectric=14,
        lattice=defect_locpot.structure.lattice,
        defect_locpot=defect_locpot,
        bulk_locpot=bulk_locpot,
        defect_frac_coords=[0.5, 0.5, 0.5],
    )

    freysoldt_summary = get_freysoldt_correction(
        q=0,
        dielectric=14,
        lattice=defect_locpot.structure.lattice,
        defect_locpot=[*map(defect_locpot.get_axis_grid, [0, 1, 2])],
        bulk_locpot=[*map(bulk_locpot.get_axis_grid, [0, 1, 2])],
        defect_frac_coords=[0.5, 0.5, 0.5],
    )
