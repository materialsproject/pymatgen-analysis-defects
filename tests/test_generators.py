import pytest

from pymatgen.analysis.defects.core import Interstitial, Substitution, Vacancy
from pymatgen.analysis.defects.generators import (
    AntiSiteGenerator,
    InterstitialGenerator,
    SubstitutionGenerator,
    VacancyGenerator,
)


def test_vacancy_generators(gan_struct):
    vacancy_generator = VacancyGenerator(gan_struct)
    for defect in vacancy_generator:
        assert isinstance(defect, Vacancy)

    vacancy_generator = VacancyGenerator(gan_struct, ["Ga"])
    cnt = 0
    for defect in vacancy_generator:
        assert isinstance(defect, Vacancy)
        cnt += 1
    assert cnt == 1

    with pytest.raises(ValueError):
        vacancy_generator = VacancyGenerator(gan_struct, ["Xe"])


def test_substitution_generators(gan_struct):
    sub_generator = SubstitutionGenerator(gan_struct, {"Ga": ["Mg", "Ca"]})
    replaced_atoms = set()
    for defect in sub_generator:
        assert isinstance(defect, Substitution)
        replaced_atoms.add(defect.site.specie.symbol)
    assert replaced_atoms == {"Mg", "Ca"}


def test_antisite_generator(gan_struct):
    anti_gen = AntiSiteGenerator(gan_struct)
    def_names = [defect.name for defect in anti_gen]
    assert sorted(def_names) == ["Ga_N", "N_Ga"]


def test_interstitial_generator(chgcar_fe3o4):
    gen = InterstitialGenerator.from_chgcar(chgcar_fe3o4, "Ga", max_avg_charge=0.5)
    cnt = 0
    for defect in gen:
        assert isinstance(defect, Interstitial)
        assert defect.site.specie.symbol == "Ga"
        cnt += 1
    assert cnt == 2
