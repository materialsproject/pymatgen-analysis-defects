from pymatgen.core.periodic_table import Specie

from pymatgen.analysis.defect.core import PeriodicSite, Substitution, Vacancy


def test_vacancy(gan_struct):
    s = gan_struct.copy()
    vac = Vacancy(s, s.sites[0])
    assert str(vac) == "Ga3+ Vacancy defect at site #0"
    assert vac.oxi_state == -3
    assert vac.get_charge_states() == [-4, -3, -2, -1, 0, 1]
    assert vac.get_multiplicity() == 2
    assert vac.get_supercell_structure().formula == "Ga63 N64"
    assert vac.name == "Va_Ga"


def test_substitution(gan_struct):
    s = gan_struct.copy()
    n_site = s.sites[3]
    assert n_site.specie.symbol == "N"
    o_site = PeriodicSite(Specie("O"), n_site.frac_coords, s.lattice)
    sub = Substitution(s, o_site)
    assert str(sub) == "O0+ subsitituted on the N3- site at at site #3"
    assert sub.oxi_state == 1
    assert sub.get_charge_states() == [-1, 0, 1, 2]
    assert sub.get_multiplicity() == 2
    sc = sub.get_supercell_structure()
    assert sc.formula == "Ga64 N63 O1"
    assert sub.name == "O_N"
