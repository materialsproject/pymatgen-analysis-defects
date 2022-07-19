pymatgen-analysis-defects
=========================

|testing| |codecov|

📄 `Full Documentation <https://materialsproject.github.io/pymatgen-analysis-defects/>`_


This package is a extension to ``pymatgen`` for performing defect analysis.
The package is designed to work with VASP inputs and output files and is meant
to be used as a namespace package extension to the main ``pymatgen`` library.
The new module has been redesigned to work closely with ``atomate2``.  While the the automation
frameworks is not required for this code to be useful, users are strongly encouraged to
to adopt the ``atomate2`` framework as it contains codified "best practices" for running defect calculations
as well as orchestrating the running of calculations and storing the results.

Previous versions of the defects code
-------------------------------------

This package is meant to replace the older ``pymatgen.analysis.defects`` modules.
The previous module was used by ``pyCDT`` code which will continue to work with version ``2022.7.8`` of ``pymatgen``.
Newer releases of ``pymatgen`` will only contain a stub in ``pymatgen.analysis.defects`` and this package must be installed separately.


.. |testing| image:: https://github.com/materialsproject/pymatgen-analysis-defects/actions/workflows/testing.yml/badge.svg?branch=main
   :target: https://github.com/materialsproject/pymatgen-analysis-defects/actions/workflows/testing.yml
.. |codecov| image:: https://codecov.io/gh/materialsproject/pymatgen-analysis-defects/branch/main/graph/badge.svg?token=FOKXRCZTXZ
   :target: https://codecov.io/gh/materialsproject/pymatgen-analysis-defects
