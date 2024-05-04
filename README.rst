pymatgen-analysis-defects
=========================

|testing| |codecov| |zenodo| |pypi|

📄 `Full Documentation <https://materialsproject.github.io/pymatgen-analysis-defects/>`_


This package is an extension to ``pymatgen`` for performing defect analysis.
The package is designed to work with VASP inputs and output files and is meant
to be used as a namespace package extension to the main ``pymatgen`` library.
The new module has been redesigned to work closely with ``atomate2``.

While the ``atomate2`` automation framework is not required for this code to be useful, users are strongly encouraged to
to adopt the ``atomate2`` framework as it contains codified "best practices" for running defect calculations
as well as orchestrating the running of calculations and storing the results.

The package serves as an object-oriented interface to defect physics and is capable of generating a list of 
non-equivalent defect objects directly from the Materials Project API.

.. code-block:: python
    from pymatgen.analysis.defects.generators import ChargeInterstitialGenerator, generate_all_native_defects
    from pymatgen.ext.matproj import MPRester
    with MPRester() as mpr:
        chgcar = mpr.get_charge_density_from_material_id("mp-804")
        
    for defect in generate_all_native_defects(chgcar):
        print(defect)

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
This package handles the energy accounting of the chemical species for the chemical
potential calculations, which determines the y-offset of the formation energy.
This package also performs finite-size corrections for the formation energy which is required
when studying charged defects in periodic simulation cells.

Defect Position
+++++++++++++++

Identification of the defect positions in a simulation cell after atomic relaxation
is not trivial since the many atoms can collectively shift in response to the creation of
the defect.
Yet the exact location of the defect is required for the calculation of finite-size corrections
as well as other physical properties.
We devised a method based on calculating a SOAP-based distortion field that can be used to
identify the defect position in a simulation cell.
Note, this method only requires the reference pristine supercell and does not need prior knowledge
of how the defect was created.

Defect Complexes
++++++++++++++++

Multiple defects can be composed into defect complexes.
The complex is can be treated as a normal defect object for subsequent analysis.

Defect Interactions
+++++++++++++++++++

Simulation of defect-photon and defect-phonon interactions under the independent particle approximation.

Previous versions of the defects code
-------------------------------------

This package replaces the older ``pymatgen.analysis.defects`` modules.
The previous module was used by ``pyCDT`` code which will continue to work with version ``2022.7.8`` of ``pymatgen``.

Contributing
------------

The source code can be downloaded from the GitHub repository at

.. code-block:: bash

    $ git clone https://github.com/materialsproject/pymatgen-analysis-defects.git

All code contributions are welcome. Please submit a pull request on GitHub.
To make maintenance easier, please use a workflow similar to the automated CI
`workflow <https://github.com/materialsproject/pymatgen-analysis-defects/blob/main/.github/workflows/testing.yml>`_.

Specifically, please make sure to run the following commands for linting:

.. code-block:: bash

    $ pip install -e .[strict]
    $ pip install -e .[dev]
    $ pre-commit install
    $ pre-commit run --all-files

And run these commands for testing:

.. code-block:: bash

    $ pip install -e .[strict]
    $ pip install -e .[tests]
    $ pytest --cov=pymatgen
    $ pytest --nbmake ./docs/source/content

For more details about what is actually installed with each of the ``pip install .[arg]`` commands, please inspect the
``pyproject.toml`` file.

Contributors
------------

* Lead developer: Dr. Jimmy-Xuan Shen
* This code contains contributions from the original defects analysis module of ``pymatgen`` from Dr. Danny Broberg and Dr. Shyam Dwaraknath.

.. |testing| image:: https://github.com/materialsproject/pymatgen-analysis-defects/actions/workflows/testing.yml/badge.svg?branch=main
   :target: https://github.com/materialsproject/pymatgen-analysis-defects/actions/workflows/testing.yml
.. |codecov| image:: https://codecov.io/gh/materialsproject/pymatgen-analysis-defects/branch/main/graph/badge.svg?token=FOKXRCZTXZ
   :target: https://codecov.io/gh/materialsproject/pymatgen-analysis-defects
.. |zenodo| image:: https://zenodo.org/badge/452872799.svg
   :target: https://zenodo.org/badge/latestdoi/452872799
.. |pypi| image:: https://badge.fury.io/py/pymatgen-analysis-defects.svg
    :target: https://badge.fury.io/py/pymatgen-analysis-defects
