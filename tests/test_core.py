import numpy as np
from pymatgen.core.periodic_table import Element, Specie

from pymatgen.analysis.defects.core import (
    Interstitial,
    PeriodicSite,
    Substitution,
    Vacancy,
)


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
    assert vac.name == "Va_Ga"
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
    assert str(inter) == "N intersitial site at at site [0.00,0.00,0.75]"
    assert inter.element_changes == {Element("N"): 1}
