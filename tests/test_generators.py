import pytest

from pymatgen.analysis.defects.core import Interstitial, Substitution, Vacancy
from pymatgen.analysis.defects.generators import (
    InterstitialGenerator,
    SubstitutionGenerator,
    VacancyGenerator,
)


def test_vacancy_generators(gan_struct):
    vacancy_generator = VacancyGenerator(gan_struct)
    for defect in vacancy_generator:
        assert isinstance(defect, Vacancy)

    vacancy_generator = VacancyGenerator(gan_struct, ["Ga"])
    for i, defect in enumerate(vacancy_generator):
        assert i < 1
        assert isinstance(defect, Vacancy)

    with pytest.raises(ValueError):
        vacancy_generator = VacancyGenerator(gan_struct, ["Xe"])


def test_substitution_generators(gan_struct):
    sub_generator = SubstitutionGenerator(gan_struct, {"Ga": "Mg"})
    replaced_atoms = set()
    for defect in sub_generator:
        assert isinstance(defect, Substitution)
        replaced_atoms.add(defect.site.specie.symbol)
    sub_generator = SubstitutionGenerator(gan_struct, {"Ga": "Ca"})
    for defect in sub_generator:
        assert isinstance(defect, Substitution)
        replaced_atoms.add(defect.site.specie.symbol)
    assert replaced_atoms == {"Mg", "Ca"}


def test_interstitial_generator(chgcar_fe3o4):
    gen = InterstitialGenerator.from_chgcar(chgcar_fe3o4, "Ga", max_avg_charge=0.5)
    for i, defect in enumerate(gen):
        assert i < 2
        assert isinstance(defect, Interstitial)
        assert defect.site.specie.symbol == "Ga"
