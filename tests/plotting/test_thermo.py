import pytest
from pymatgen.analysis.defects.plotting.thermo import plot_formation_energy_diagrams, plot_chempot_2d
from pymatgen.core import Element

def test_fed_plot(basic_fed):
    fig = plot_formation_energy_diagrams([basic_fed])
    assert {d_.name for d_ in fig.data} == {'Mg_Ga', 'Mg_Ga:slope'}

def test_chempot_plot(basic_fed):
    plot_chempot_2d(basic_fed, x_element=Element("Mg"), y_element=Element("Ga"))



