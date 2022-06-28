import pytest


def test_ccd(test_dir):
    from pymatgen.analysis.defects2.ccd import ConfigurationCoordinateDiagram

    ccd = ConfigurationCoordinateDiagram(
        charge_gs=0,
        charge_es=1,
        dQ=1.0,
        dE=2,
        Q_gs=(-0.2, 0.0, 0.2),
        Q_es=(0.8, 1.0, 1.2),
        energies_es=(1.2, 1.0, 1.2),
        energies_gs=(1.2, 1.0, 1.2),
    )
    # check that plot runs without error don't show
    ccd.plot(show=False)
    assert ccd.omega_es == pytest.approx(ccd.omega_gs)
