# pymatgen-analysis-defects

[![testing](https://github.com/materialsproject/pymatgen-analysis-defects/actions/workflows/testing.yml/badge.svg)](https://github.com/materialsproject/pymatgen-analysis-defects/actions/workflows/testing.yml)


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


## Automated code linting

Pymatgen-analysis-defects uses `pre-commit` to automatically lint the code.
This is a good way to ensure that the code is clean and well-formatted.
To install pre-commit, run the following command:
```
pip install pre-commit==[VERSION]
```
where [VERSION] is the version of `pre-commit` designated in the `requirements-ci.txt` file.
Once installed, run the following command to install the pre-commit hooks:
```
pre-commit install
```
After installing the pre-commit hooks, you can run the following command to lint the code:
```
pre-commit run --all-files
```
Or simply commit the changes using `git commit` to automatically lint the code on any file that was modified.
