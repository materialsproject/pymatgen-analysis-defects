from collections import defaultdict
from pathlib import Path

import pytest
from monty.serialization import loadfn
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Specie
from pymatgen.io.vasp import Locpot, Vasprun

from pymatgen.analysis.defects2.core import PeriodicSite, Substitution
from pymatgen.analysis.defects2.thermo import DefectEntry


@pytest.fixture(scope="session")
def test_dir():
    module_dir = Path(__file__).resolve().parent
    test_dir = module_dir / "test_files"
    return test_dir.resolve()


@pytest.fixture(scope="session")
def gan_struct(test_dir):
    return Structure.from_file(test_dir / "GaN.vasp")


@pytest.fixture(scope="session")
def stable_entries_Mg_Ga_N(test_dir):
    return loadfn(test_dir / "stable_entries_Mg_Ga_N.json")


@pytest.fixture(scope="session")
def defect_Mg_Ga(gan_struct):
    ga_site = gan_struct[0]
    mg_site = PeriodicSite(Specie("Mg"), ga_site.frac_coords, gan_struct.lattice)
    return Substitution(gan_struct, mg_site)


@pytest.fixture(scope="session")
def data_Mg_Ga(test_dir):
    """
    Get the data in the following format:
    {
        "bulk_sc": {
            "vasp_run": Vasprun,
            "locpot": Locpot,
        },
        "q=1": {
            "vasp_run": Vasprun,
            "locpot": Locpot,
        },
        ...
    }
    """
    root_dir = test_dir / "Mg_Ga"
    data = defaultdict(dict)
    for fold in root_dir.glob("./*"):
        if not fold.is_dir():
            continue
        data[fold.name] = {
            "vasprun": Vasprun(fold / "vasprun.xml.gz"),
            "locpot": Locpot.from_file(fold / "LOCPOT.gz"),
        }
    return data


@pytest.fixture(scope="session")
def defect_entries_Mg_Ga(data_Mg_Ga, defect_Mg_Ga):
    bulk_locpot = data_Mg_Ga["bulk_sc"]["locpot"]

    def get_data(q):
        computed_entry = data_Mg_Ga[f"q={q}"]["vasprun"].get_computed_entry(inc_structure=True)
        defect_locpot = data_Mg_Ga[f"q={q}"]["locpot"]

        def_entry = DefectEntry(defect=defect_Mg_Ga, charge_state=q, sc_entry=computed_entry, dielectric=14)
        plot_data = def_entry.get_freysoldt_correction(defect_locpot=defect_locpot, bulk_locpot=bulk_locpot)
        return def_entry, plot_data

    defect_entries = dict()
    plot_data = dict()
    for qq in [-2, -1, 0, 1]:
        defect_entry, p_data = get_data(qq)
        defect_entries[qq] = defect_entry
        plot_data[qq] = p_data
    return defect_entries, plot_data
