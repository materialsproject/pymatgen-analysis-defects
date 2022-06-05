from collections import defaultdict
from pathlib import Path

import pytest
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Specie
from pymatgen.io.vasp import Locpot, Vasprun

from pymatgen.analysis.defect.core import PeriodicSite, Substitution


@pytest.fixture(scope="session")
def test_dir():
    module_dir = Path(__file__).resolve().parent
    test_dir = module_dir / "test_files"
    return test_dir.resolve()


@pytest.fixture(scope="session")
def gan_struct(test_dir):
    return Structure.from_file(test_dir / "GaN.vasp")


@pytest.fixture(scope="session")
def defect_Mg_Ga(gan_struct):
    ga_site = gan_struct[0]
    mg_site = PeriodicSite(Specie("Mg"), ga_site.frac_coords, gan_struct.lattice)
    return Substitution(gan_struct, mg_site)


@pytest.fixture(scope="session")
def vasp_Mg_Ga(test_dir):
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
