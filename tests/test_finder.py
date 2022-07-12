def test_defect_finder(test_dir):
    from pymatgen.core import IStructure

    from pymatgen.analysis.defects.finder import DefectSiteFinder

    base = IStructure.from_file(test_dir / "GaN.vasp")

    # Vacancy
    sc = base * [2, 2, 2]
    frac_pos_rm = sc.sites[9].frac_coords
    sc.remove_sites([9])
    finder = DefectSiteFinder()
    frac_pos_guess = finder.get_native_defect_position(sc, base)
    dist, _ = sc.lattice.get_distance_and_image(frac_pos_guess, frac_pos_rm)
    assert dist < 0.5

    # Interstitial
    sc = base * [2, 2, 2]
    frac_pos_insert = [0.666665, 0.333335, 0.31206]
    sc.insert(0, "Ga", frac_pos_insert)
    frac_pos_guess = finder.get_native_defect_position(sc, base)
    dist, _ = sc.lattice.get_distance_and_image(frac_pos_guess, frac_pos_insert)
    assert dist < 0.5

    # Anti-site
    sc = base * [2, 2, 2]
    Ga_pos = sc.sites[12].frac_coords
    N_pos = sc.sites[16].frac_coords
    dist, _ = sc.lattice.get_distance_and_image(Ga_pos, N_pos)
    assert dist < 2
    # swapping two sites that are close to each other
    sc.remove_sites([16])
    sc.remove_sites([12])
    # have the distort slightly to the midpoint
    mid_point = (N_pos + Ga_pos) / 2
    sc.insert(0, "N", 0.99 * Ga_pos + 0.01 * mid_point)
    sc.insert(0, "Ga", 0.99 * N_pos + 0.01 * mid_point)

    frac_pos_guess = finder.get_native_defect_position(sc, base)
    dist, _ = sc.lattice.get_distance_and_image(frac_pos_guess, mid_point)
    assert dist < 0.5
