"""Constants used in the Defects module."""


import numpy as np
from scipy import constants

HBAR_EV = constants.physical_constants["Planck constant over 2 pi in eV s"][0]
HBAR_J = constants.physical_constants["Planck constant over 2 pi"][0]
EV2J = constants.e  # 1 eV in Joules
AMU2KG = constants.physical_constants["atomic mass constant"][0]
ANGS2M = 1e-10  # angstrom in meters
KB = constants.physical_constants["Boltzmann constant in eV/K"][0]

AU2ANG = constants.physical_constants["atomic unit of length"][0] / 1e-10
RYD2EV = constants.physical_constants["Rydberg constant times hc in eV"][0]
EDEPS = 4 * np.pi * 2 * RYD2EV * AU2ANG  # exactly the same as VASP

LOOKUP_TABLE = np.array(
    [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
    ],
    dtype=np.double,
)
