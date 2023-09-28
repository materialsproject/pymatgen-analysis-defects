---
title: 'pymatgen-analysis-defect: A Python package for analyzing point defects in crystalline materials'

tags:
  - python
  - materials science
  - point defects
  - finite-size corrections
  - database building
authors:
  - name: Jimmy-Xuan Shen
    orcid: 0000-0002-2743-7531
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Joel Varley
    affiliation: "1"
affiliations:
  - name: Lawrence Livermore National Laboratory, Livermore, California 94550, United States
    index: 1
date: August 2023
bibliography: paper.bib
---

# Summary

Point defects can often determine the properties of semiconductor and optoelectronic materials.
Simulation of point defects is one of the most complex workflows in computational materials science, involving extensive pre- and post-processing of the structural and electronic structure data[@CGWalle_defects_RMP].
Multiple software packages exist to automate the simulation of point defects [@Broberg2018,@Kumagai2021,@Huang2022,], however, there is a lack of a code that focuses on:

1. Integration but not insistence on standardized high-throughput workflow frameworks.
2. Building large, persistent databases of point defects that are extensible to new calculations over time.

Since the combinatorics of point defects in crystalline materials can be extremely large, it is important to have a software package that can be easily integrated into high-throughput workflows to manage these complex calculations.
However, most users of a defect analysis packages will not need to rung thousands of calcualaitons so it is important to have code focused purely on the defect analysis and relegate the high-throughput workflow aspect to a separate package.
Additionally, a well-known problem in the simulation of point defects is the fact that current structure optimization techniques can miss the ground state structure based on the initial guess in a sizable minority of cases, so the ability to easily re-visit and re-optimize structures is crucial to building a reliable database of point defects.
Towards that end, we have developed a Python package, `pymatgen-analysis-defect`, and integrated it with the popular `atomate2` workflow framework to provide a complete set of tools for simulating, analyzing, and managing the results of point defect calculations.

Since the ability to revisit calculations is crucial to building a reliable database, but user tagging of calculations is inconsistent, especially in a high-throughput context, we have codified a structure-only definition of point defects that can be used to aggregate the results of multiple calculations of the same defect.
This allows for the creation of a database of point defects that can be easily extended to new calculations over time.
In addition to the focus on database building, we have also implemented several tools for analyzing quantum recombination in defects, these include:

1. Obtaining the chemical potential contribution to the defect formation energy without explicit calculations of the competing phases.
2. Obtaining the Freysoldt finite-size correction without user intervention.
3. Calculation of the optical transition between states under the independent-particle approximation.
4. Calculation of the non-radiative recombination using the `nonrad` code [@turiansky_nonrad_2021].

Details of the implementation and tutorials for using the different parts of the package are provided at:

https://materialsproject.github.io/pymatgen-analysis-defects/intro.html

## Defect Definition

A core feature of `pymatgen-analysis-defect` is the ability to define a point defects automatically.
While symmetry analysis on the atomic structure alone is usually enough to define the distinct substitutional and vacancy defects, we found that the electronic charge density was the most effective at placing the interstitial defect at symmetry-inequivalent positions.
A basic example of creating a full list of defects is shown below:

```python
from pymatgen.analysis.defects.generators import generate_all_native_defects
from pymatgen.ext.matproj import MPRester

with MPRester() as mpr:
    chgcar = mpr.get_charge_density_from_material_id("mp-804")

defects = []
for defect in generate_all_native_defects(chgcar):
    print(defect)
    defects.append(defect)
```

```
Ga Vacancy defect at site #0
N Vacancy defect at site #2
N subsitituted on the Ga site at at site #0
Ga subsitituted on the N site at at site #2
Ga intersitial site at [0.00,0.00,0.20]
Ga intersitial site at [0.35,0.65,0.69]
N intersitial site at [0.00,0.00,0.20]
N intersitial site at [0.35,0.65,0.69]
```

In the code above, we query the materials project database for the charge density object which contains the information about the bulk structure as well as the electronic charge density.
Using the `generate_all_native_defects` function, we can generate a list of all of the native point defects for this structure.
![Figure 1. Defect generation.](fig1.png)

## Defect Simulation Workflow

A basic example of integration with the `atomate2` workflow framework is provided below:

```python
from atomate2.vasp.flows.defect import FormationEnergyMaker
from jobflow import Flow
from pymatgen.analysis.defects.generators import generate_all_native_defects
from pymatgen.ext.matproj import MPRester

with MPRester() as mpr:
    chgcar = mpr.get_charge_density_from_material_id("mp-804")


maker = FormationEnergyMaker()
jobs = []
for defect in generate_all_native_defects(chgcar):
    jobs.append(maker.make(defect))
flow = Flow(jobs)
```

The code above will generate a `Flow` object that contains all of the instructions to dynamically create all of the required defect calculations, which can be sent to the job manager on HPC to be executed.


# Statement of need

Due to the large simulation cell and the higher-cost density functionals required for defect simulations, the computational cost of defect calculations is often orders of magnitude higher than that of bulk calculations.
As such, managing and curating the results of the defect calculations generated by a single user has to potential to save a significant amount of computational resources.
Moreover, eventually building a high-quality, persistent defects database will significantly reduce the computational cost of defect calculations for the entire community.
The present software package is designed to facilitate both of these goals.

# Acknowledgements

...
