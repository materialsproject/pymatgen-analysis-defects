# pymatgen-analysis-defects

[![testing](https://github.com/materialsproject/pymatgen-analysis-defects/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/materialsproject/pymatgen-analysis-defects/actions/workflows/testing.yml)
[![codecov](https://codecov.io/gh/materialsproject/pymatgen-analysis-defects/branch/main/graph/badge.svg?token=FOKXRCZTXZ)](https://codecov.io/gh/materialsproject/pymatgen-analysis-defects)

This package is a collection of tools for analyzing defects in materials.
It is meant to replace the older `pymatgen.analysis.defects` modules.
The code is currently installed at `pymatgen.analysis.defects2`.
but will be moved to `pymatgen.analysis.defects` in the future.

The modules provided by this package are summarized below:

| Module name   |                        Functionality                         |
|---------------|:------------------------------------------------------------:|
| `ccd`         |             configuration-coordination diagrams.             |
| `core`        | abstract definition of a defect (unit cell) + (defect site). |
| `corrections` |         corrections to the defect formation energy.          |
| `finder`      |       identify the position of defects automatically.        |
| `generators`  |     analyze bulk crystal symmetry to generator defects.      |
| `supercells`  |   code to generate supercells for simulating the defects.    |
| `thermo`      |            formation energy diagram definitions.             |
| `utils`       |                   miscellaneous utilities.                   |
