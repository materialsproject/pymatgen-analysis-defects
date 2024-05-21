from collections import defaultdict
from pathlib import Path

import pytest
from monty.serialization import loadfn
from pymatgen.analysis.defects.core import PeriodicSite, Substitution
from pymatgen.analysis.defects.thermo import DefectEntry, PhaseDiagram
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Specie
from pymatgen.io.vasp.outputs import WSWQ, Chgcar, Locpot, Procar, Vasprun


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
    """Get the data in the following format:
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
def defect_entries_and_plot_data_Mg_Ga(data_Mg_Ga, defect_Mg_Ga):
    bulk_locpot = data_Mg_Ga["bulk_sc"]["locpot"]

    def get_data(q):
        computed_entry = data_Mg_Ga[f"q={q}"]["vasprun"].get_computed_entry(
            inc_structure=True
        )
        defect_locpot = data_Mg_Ga[f"q={q}"]["locpot"]

        def_entry = DefectEntry(
            defect=defect_Mg_Ga, charge_state=q, sc_entry=computed_entry
        )
        frey_summary = def_entry.get_freysoldt_correction(
            defect_locpot=defect_locpot, bulk_locpot=bulk_locpot, dielectric=14
        )
        return def_entry, frey_summary

    defect_entries = dict()
    plot_data = dict()
    for qq in [-2, -1, 0, 1]:
        defect_entry, frey_summary = get_data(qq)
        defect_entries[qq] = defect_entry
        plot_data[qq] = frey_summary.metadata["plot_data"]
    return defect_entries, plot_data


@pytest.fixture(scope="session")
def chgcar_fe3o4(test_dir):
    return Chgcar.from_file(test_dir / "CHGCAR.Fe3O4.vasp")


@pytest.fixture(scope="session")
def v_ga(test_dir):
    res = dict()
    for q1, q2 in [(0, -1), (-1, 0)]:
        ccd_dir = test_dir / f"v_Ga/ccd_{q1}_{q2}"
        vaspruns = [Vasprun(ccd_dir / f"{i}/vasprun.xml") for i in [0, 1, 2]]
        wswq_dir = ccd_dir / "wswqs"
        wswq_files = [f for f in wswq_dir.glob("WSWQ*")]
        wswq_files.sort(
            key=lambda x: int(x.name.split(".")[1])
        )  # does stem work for non-zipped files?
        wswqs = [WSWQ.from_file(f) for f in wswq_files]
        # wswqs = [WSWQ.from_file(ccd_dir / "wswqs" / f"WSWQ.{i}.gz") for i in [0, 1, 2]]
        res[(q1, q2)] = {
            "vaspruns": vaspruns,
            "procar": Procar(ccd_dir / "1/PROCAR"),
            "wswqs": wswqs,
        }
    return res


@pytest.fixture(scope="session")
def v_N_GaN(test_dir):
    """More complex."""
    bulk_locpot = Locpot.from_file(test_dir / "v_N_GaN/bulk/LOCPOT.gz")
    return {
        "bulk_locpot": bulk_locpot,
        "defect_locpots": {
            -1: Locpot.from_file(test_dir / "v_N_GaN/q=-1/LOCPOT.gz"),
            0: Locpot.from_file(test_dir / "v_N_GaN/q=0/LOCPOT.gz"),
            1: Locpot.from_file(test_dir / "v_N_GaN/q=1/LOCPOT.gz"),
            2: Locpot.from_file(test_dir / "v_N_GaN/q=2/LOCPOT.gz"),
        },
    }


@pytest.fixture(scope="module")
def formation_energy_diagram(
    data_Mg_Ga, defect_entries_and_plot_data_Mg_Ga, stable_entries_Mg_Ga_N
):
    bulk_vasprun = data_Mg_Ga["bulk_sc"]["vasprun"]
    bulk_bs = bulk_vasprun.get_band_structure()
    vbm = bulk_bs.get_vbm()["energy"]
    bulk_entry = bulk_vasprun.get_computed_entry(inc_structure=False)
    defect_entries, _ = defect_entries_and_plot_data_Mg_Ga

    def_ent_list = list(defect_entries.values())
    # test the constructor with materials project phase diagram
    atomic_entries = list(
        filter(lambda x: len(x.composition.elements) == 1, stable_entries_Mg_Ga_N)
    )
    pd = PhaseDiagram(stable_entries_Mg_Ga_N)
    # test the constructor with atomic entries
    # this is the one we will use for the rest of the tests
    fed = FormationEnergyDiagram.with_atomic_entries(
        defect_entries=def_ent_list,
        atomic_entries=atomic_entries,
        vbm=vbm,
        inc_inf_values=False,
        phase_diagram=pd,
        bulk_entry=bulk_entry,
    )
    assert len(fed.chempot_limits) == 3

    # dataframe conversion
    df = fed.as_dataframe()
    assert df.shape == (4, 5)

    # test that you can get the Ga-rich chempot
    cp = fed.get_chempots(rich_element=Element("Ga"))
    assert cp[Element("Ga")] == pytest.approx(0, abs=1e-2)

    return fed
