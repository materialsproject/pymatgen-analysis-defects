from pymatgen.analysis.defect.thermo import get_lower_envelope, get_transitions


def test_lower_envelope():
    # Test the lower envelope and transition code with a simple example
    lines = [[4, 12], [-1, 3], [-5, 4], [-2, 1], [3, 8], [-4, 14], [2, 12], [3, 8]]
    lower_envelope_ref = [(4, 12), (3, 8), (-2, 1), (-5, 4)]  # answer from visual inspection (ordered)
    transitions_ref = [(-4, -4), (-1.4, 3.8), (1, -1)]
    lower_envelope = get_lower_envelope(lines)
    assert lower_envelope == lower_envelope_ref
    assert get_transitions(lower_envelope) == transitions_ref


def test_free_energy(vasp_Mg_Ga, defect_Mg_Ga):
    data = vasp_Mg_Ga

    bulk_entry = data["bulk_sc"]["vasprun"].get_computed_entry(inc_structure=True)
    data["bulk_sc"]["locpot"]
