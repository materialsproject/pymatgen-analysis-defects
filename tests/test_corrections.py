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
    assert freysoldt_summary.correction_energy == pytest.approx(0, abs=1e-4)

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


def test_kumagai(test_dir):
    from pymatgen.analysis.defects.corrections.kumagai import (
        get_efnv_correction,
        read_vasp_output,
    )

    calc_bulk = read_vasp_output(test_dir / "Mg_Ga" / "bulk_sc")
    calc_defect = read_vasp_output(test_dir / "Mg_Ga" / "q=0")

    sd = calc_defect.structure.copy()
    sd.add_site_property("potential", calc_defect.potentials)

    sb = calc_bulk.structure.copy()
    sb.add_site_property("potential", calc_bulk.potentials)
    res = get_efnv_correction(
        0, sd, sb, dielectric_tensor=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    assert res == pytest.approx(0, abs=1e-4)
