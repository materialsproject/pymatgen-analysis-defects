# pymatgen-analysis-defects

[![testing](https://github.com/materialsproject/pymatgen-analysis-defects/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/materialsproject/pymatgen-analysis-defects/actions/workflows/testing.yml)
[![codecov](https://codecov.io/gh/materialsproject/pymatgen-analysis-defects/branch/main/graph/badge.svg?token=FOKXRCZTXZ)](https://codecov.io/gh/materialsproject/pymatgen-analysis-defects)

This package is a collection of tools for analyzing defects in materials.
It currently functions as a namespace package add-on to pymatgen's existing defect analysis tools.
Once installed the additional modules will still appear under the
`pymatgen.analysis.defects` namespace.

The additional modules provided by this package are summarized below:

| Module name |                  Funtionality                   |
|-------------|:-----------------------------------------------:|
| `ccd`       |      configuration-coordination diagrams.       |
| `finder`    | identify the position of defects automatically. |

Note that each module is a sub-package of the `pymatgen.analysis.defects` namespace, and must be imported
as `pymatgen.analysis.defects.[module_name]`.
