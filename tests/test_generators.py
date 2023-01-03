import pytest

from pymatgen.analysis.defects.core import Interstitial, Substitution, Vacancy
from pymatgen.analysis.defects.generators import (
    AntiSiteGenerator,
    ChargeInterstitialGenerator,
    InterstitialGenerator,
    SubstitutionGenerator,
    VacancyGenerator,
    VoronoiInterstitialGenerator,
)


def test_vacancy_generators(gan_struct):
    vacancy_generator = VacancyGenerator().get_defects(gan_struct)
    for defect in vacancy_generator:
        assert isinstance(defect, Vacancy)

    vacancy_generator = VacancyGenerator().get_defects(gan_struct, ["Ga"])
    cnt = 0
    for defect in vacancy_generator:
        assert isinstance(defect, Vacancy)
        cnt += 1
    assert cnt == 1

    with pytest.raises(ValueError):
        vacancy_generator = list(
            VacancyGenerator().get_defects(gan_struct, rm_species=["Xe"])
        )


def test_substitution_generators(gan_struct):
    sub_generator = SubstitutionGenerator().get_defects(
        gan_struct, {"Ga": ["Mg", "Ca"]}
    )
    replaced_atoms = set()
    for defect in sub_generator:
        assert isinstance(defect, Substitution)
        replaced_atoms.add(defect.site.specie.symbol)
    assert replaced_atoms == {"Mg", "Ca"}

    sub_generator = SubstitutionGenerator().get_defects(gan_struct, {"Ga": "Mg"})
    replaced_atoms = set()
    for defect in sub_generator:
        assert isinstance(defect, Substitution)
        replaced_atoms.add(defect.site.specie.symbol)
    assert replaced_atoms == {
        "Mg",
    }


def test_antisite_generator(gan_struct):
    anti_gen = AntiSiteGenerator().get_defects(gan_struct)
    def_names = [defect.name for defect in anti_gen]
    assert sorted(def_names) == ["Ga_N", "N_Ga"]


def test_interstitial_generator(gan_struct):
    gen = InterstitialGenerator().get_defects(
        gan_struct, insertions={"Mg": [[0, 0, 0]]}
    )
    l_gen = list(gen)
    assert len(l_gen) == 1
    assert str(l_gen[0]) == "Mg intersitial site at [0.00,0.00,0.00]"

    bad_site = [0.667, 0.333, 0.875]
    gen = InterstitialGenerator().get_defects(
        gan_struct, insertions={"Mg": [[0, 0, 0], bad_site]}
    )
    l_gen = list(gen)
    assert len(l_gen) == 1


def test_charge_interstitial_generator(chgcar_fe3o4):
    gen = ChargeInterstitialGenerator().get_defects(chgcar_fe3o4, {"Ga"})
    cnt = 0
    for defect in gen:
        assert isinstance(defect, Interstitial)
        assert defect.site.specie.symbol == "Ga"
        cnt += 1
    assert cnt == 2


def test_voronoi_interstitial_generator(chgcar_fe3o4):
    gen = VoronoiInterstitialGenerator().get_defects(chgcar_fe3o4.structure, {"Li"})
    cnt = 0
    for defect in gen:
        assert isinstance(defect, Interstitial)
        assert defect.site.specie.symbol == "Li"
        cnt += 1
    assert cnt == 4
