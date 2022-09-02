pymatgen-analysis-defects
=========================

|testing| |codecov|

📄 `Full Documentation <https://materialsproject.github.io/pymatgen-analysis-defects/>`_


This package is an extension to ``pymatgen`` for performing defect analysis.
The package is designed to work with VASP inputs and output files and is meant
to be used as a namespace package extension to the main ``pymatgen`` library.
The new module has been redesigned to work closely with ``atomate2``.

While the ``atomate2`` automation framework is not required for this code to be useful, users are strongly encouraged to
to adopt the ``atomate2`` framework as it contains codified "best practices" for running defect calculations
as well as orchestrating the running of calculations and storing the results.


Non-exhaustive list of features:
--------------------------------

Reproducible definition of defects
++++++++++++++++++++++++++++++++++

Defects are defined based on the physical concept they represent,
independent of the calculation details such as simulation cell size.
As an example, a Vacancy defect is defined by the primitive cell of the
pristine material plus a single site that represents the vacancy site in
the unit cell.


Formation energy calculations
+++++++++++++++++++++++++++++

The formation energy diagram is a powerful tool for understanding the
thermodynamics of defects. This package provides a simple interface for
calculating the formation energy diagram from first-principles results.

The package handles the energy accounting of the chemical species for the chemical
potential calcultions, which determines the y-offset of the formation energy.


Previous versions of the defects code
-------------------------------------

This package replaces the older ``pymatgen.analysis.defects`` modules.
The previous module was used by ``pyCDT`` code which will continue to work with version ``2022.7.8`` of ``pymatgen``.

Contributor
-----------

* Lead developer: Dr. Jimmy-Xuan Shen
* This code is a re-write of the defects analysis module of ``pymatgen`` from Dr. Danny Broberg and Dr. Shyam Dwaraknath.


.. |testing| image:: https://github.com/materialsproject/pymatgen-analysis-defects/actions/workflows/testing.yml/badge.svg?branch=main
   :target: https://github.com/materialsproject/pymatgen-analysis-defects/actions/workflows/testing.yml
.. |codecov| image:: https://codecov.io/gh/materialsproject/pymatgen-analysis-defects/branch/main/graph/badge.svg?token=FOKXRCZTXZ
   :target: https://codecov.io/gh/materialsproject/pymatgen-analysis-defects
