import numpy as np
from pymatgen.core.periodic_table import Element, Specie

from pymatgen.analysis.defects.core import (
    Adsorbate,
    DefectComplex,
    Interstitial,
    PeriodicSite,
    Substitution,
    Vacancy,
)
from pymatgen.analysis.defects.finder import DefectSiteFinder


def test_vacancy(gan_struct):
    s = gan_struct.copy()
    vac = Vacancy(s, s.sites[0])
    vac2 = Vacancy(s, s.sites[1])
    assert vac == vac2  # symmetry equivalent sites
    assert str(vac) == "Ga Vacancy defect at site #0"
    assert vac.oxi_state == -3
    assert vac.get_charge_states() == [-4, -3, -2, -1, 0, 1]
    assert vac.get_multiplicity() == 2
    assert vac.get_supercell_structure().formula == "Ga63 N64"
    assert vac.name == "v_Ga"
    assert vac == vac
    assert vac.element_changes == {Element("Ga"): -1}


def test_substitution(gan_struct):
    s = gan_struct.copy()
    n_site = s.sites[3]
    assert n_site.specie.symbol == "N"
    o_site = PeriodicSite(Specie("O"), n_site.frac_coords, s.lattice)
    o_site2 = PeriodicSite(Specie("O"), s.sites[2].frac_coords, s.lattice)
    sub = Substitution(s, o_site)
    sub2 = Substitution(s, o_site2)
    assert sub == sub2  # symmetry equivalent sites
    assert str(sub) == "O subsitituted on the N site at at site #3"
    assert sub.oxi_state == 1
    assert sub.get_charge_states() == [-1, 0, 1, 2]
    assert sub.get_multiplicity() == 2
    sc = sub.get_supercell_structure()
    assert sc.formula == "Ga64 N63 O1"
    assert sub.name == "O_N"
    assert sub == sub
    assert sub.element_changes == {Element("N"): -1, Element("O"): 1}

    # test supercell with locking
    sc_locked = sub.get_supercell_structure(relax_radius=5.0)
    free_sites = [
        i
        for i, site in enumerate(sc_locked)
        if site.properties["selective_dynamics"][0]
    ]

    finder = DefectSiteFinder()
    fpos = finder.get_defect_fpos(sc_locked, sub.structure)
    cpos = sc_locked.lattice.get_cartesian_coords(fpos)
    free_sites_ref = sc_locked.get_sites_in_sphere(cpos, 5.0, include_index=True)
    free_sites_ref = [site.index for site in free_sites_ref]
    free_sites_union = set(free_sites_ref) | set(free_sites)
    free_sites_intersection = set(free_sites_ref) & set(free_sites)
    assert len(free_sites_intersection) / len(free_sites_union) == 1.0

    # test perturbation
    sc_locked = sub.get_supercell_structure(relax_radius=5.0, perturb=0.0)
    free_sites_ref2 = sc_locked.get_sites_in_sphere(cpos, 5.0, include_index=True)
    free_sites_ref2 = [site.index for site in free_sites_ref2]
    assert set(free_sites_ref2) == set(free_sites_ref)

    # test for user defined charge
    dd = sub.as_dict()
    dd["user_charges"] = [-100, 102]
    sub_ = Substitution.from_dict(dd)
    assert sub_.get_charge_states() == [-100, 102]

    dd["user_charges"] = []  # empty list == None => use oxidation state info
    sub_ = Substitution.from_dict(dd)
    assert sub_.get_charge_states() == [-1, 0, 1, 2]


def test_interstitial(gan_struct):
    s = gan_struct.copy()
    inter_fpos = [0, 0, 0.75]
    n_site = PeriodicSite(Specie("N"), inter_fpos, s.lattice)
    inter = Interstitial(s, n_site)
    assert inter.oxi_state == 3
    assert inter.get_charge_states() == [-1, 0, 1, 2, 3, 4]
    assert np.allclose(inter.defect_structure[0].frac_coords, inter_fpos)
    sc = inter.get_supercell_structure()
    assert sc.formula == "Ga64 N65"
    assert inter.name == "N_i"
    assert str(inter) == "N intersitial site at [0.00,0.00,0.75]"
    assert inter.element_changes == {Element("N"): 1}


def test_adsorbate(gan_struct):
    s = gan_struct.copy()
    ads_fpos = [0, 0, 0.75]
    n_site = PeriodicSite(Specie("N"), ads_fpos, s.lattice)
    ads = Adsorbate(s, n_site)
    assert ads.name == "N_{ads}"
    assert str(ads) == "N adsorbate site at [0.00,0.00,0.75]"


def test_complex(gan_struct):
    s = gan_struct.copy()
    o_site = PeriodicSite(Specie("O"), s[3].frac_coords, s.lattice)
    sub = Substitution(s, o_site)  # O substituted on N site
    vac = Vacancy(s, s.sites[0])  # Ga vacancy
    inter = Interstitial(
        s, PeriodicSite(Specie("H"), [0, 0, 0.75], s.lattice)
    )  # H interstitial
    dc = DefectComplex([sub, vac])
    assert dc.name == "O_N+v_Ga"
    sc_struct = dc.get_supercell_structure()
    assert sc_struct.formula == "Ga63 N63 O1"
    dc.oxi_state == sub.oxi_state + vac.oxi_state
    dc.element_changes == {Element("Ga"): -1, Element("N"): -1, Element("O"): 1}
    dc.defect_structure.formula == "Ga1 N1 O1"

    dc2 = DefectComplex([sub, vac, inter])
    assert dc2.name == "O_N+v_Ga+H_i"
    sc_struct = dc2.get_supercell_structure(dummy_species="Xe")
    assert sc_struct.formula == "Ga63 H1 Xe3 N63 O1"  # Three defects three dummies
