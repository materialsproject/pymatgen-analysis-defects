from pymatgen.analysis.defect.core import Substitution, Vacancy
from pymatgen.analysis.defect.generators import SubstitutionGenerator, VacancyGenerator


def test_generators(gan_struct):
    # Vacancy
    vacancy_generator = VacancyGenerator(gan_struct)
    for defect in vacancy_generator:
        assert isinstance(defect, Vacancy)

    # Substitution
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
