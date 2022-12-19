#!/usr/bin/env python
# coding: utf-8

# # Defining Defects
# 
# A persistent challenge with organizing computational defect data is the ambiguity with which a defect simulation is defined.
# The standard practice is to simulate the isolated defects by using a larger simulation to create an isolated defect and then using charge-state corrections to approximate the properties of a defect in the dilute limit.
# This means that the same defect can be simulated with different simulation cells.
# Ideally, if you want to build a computational defects database that lasts many years, you cannot rely on user-supplied provenance to aggregate the data.
# You must have the external method for tracking whether two calculations are for the same defect.
# 
# <div class="alert alert-block alert-info"><b>Note:</b> This is only important for large database building. For an example of this please see the Materials Project battery database which only relies on structure matching to arrive at the list of insertion electrodes.
# </div>
# 
# A core concept in this package is that a defect is defined independently of the simulation cell.
# All of the information about which defect we are simulating is captured by the `Defect` object.
# A point defect is defined by the Wigner-Seitz unit cell representation of the bulk material stored as a `structure` attribute and a `site` attribute that indicates where the defect is in the unit cell.
# Different kinds of point defects all subclass the `Defect` objects which gives easy access to functions such as generating a cubic simulation supercell.
# As along as the database or any algorithm keeps track of this `Defect` object, you can just use simple structure matching to find out if two simulations represent the same defect.
# 

# ## Basic Example Using GaN

# In[1]:


from pathlib import Path

TEST_FILES = Path("../../../tests/test_files")


# In[2]:


from pymatgen.analysis.defects.core import DefectComplex, Substitution, Vacancy
from pymatgen.core import PeriodicSite, Species, Structure

bulk = Structure.from_file(TEST_FILES / "GaN.vasp")
if (
    bulk.lattice == bulk.get_primitive_structure().lattice
):  # check that you have the primitive structure
    print("The bulk unit cell is the unique primitive WS cell")


# Since the two Ga sites are equivalent the Mg substitution we derive from both should be equivalent.
# 

# In[3]:


ga_site0 = bulk.sites[0]
ga_site1 = bulk.sites[1]
n_site0 = bulk.sites[2]
n_site1 = bulk.sites[3]

mg_site0 = PeriodicSite(Species("Mg"), ga_site0.frac_coords, bulk.lattice)
mg_site1 = PeriodicSite(Species("Mg"), ga_site1.frac_coords, bulk.lattice)


# In[4]:


mg_ga_defect0 = Substitution(structure=bulk, site=mg_site0)
mg_ga_defect1 = Substitution(structure=bulk, site=mg_site1)
if mg_ga_defect0 == mg_ga_defect1:
    print("The two Mg_Ga defects are equivalent.")


# Equivalence here is determined using the standard StructureMatcher settings.
# 
# ```python
#     def __eq__(self, __o: object) -> bool:
#         """Equality operator."""
#         if not isinstance(__o, Defect):
#             raise TypeError("Can only compare Defects to Defects")
#         sm = StructureMatcher(comparator=ElementComparator())
#         return sm.fit(self.defect_structure, __o.defect_structure)
# ```
# 
# If you are in the situation where your lattice parameters have changed overtime (i.e. by inclusion of different functionals) you might consider using more custom maching between the defect.
# 
# Since the defect equivalence is only determined by the `defect_structure` field, all defects can be compared using `__eq__`.
# 

# In[5]:


vac_defect0 = Vacancy(structure=bulk, site=mg_site0)
vac_defect1 = Vacancy(structure=bulk, site=n_site0)
vac_defect2 = Vacancy(structure=bulk, site=n_site1)
if vac_defect0 != vac_defect1:
    print(
        f"The two vacancies {vac_defect0.name} and {vac_defect1.name} are not equivalent."
    )

if vac_defect2 == vac_defect1:
    print(
        f"The two vacancies {vac_defect2.name} and {vac_defect1.name} are equivalent."
    )


# ## Defining defect complexes
# 
# Defining defect complexes can be done by providing a list of defect objects generated using the same pristine structure.
# 

# In[6]:


def_comp0 = DefectComplex(defects=[mg_ga_defect0, vac_defect1])
def_comp1 = DefectComplex(defects=[mg_ga_defect1, vac_defect1])
def_comp2 = DefectComplex(defects=[mg_ga_defect1, vac_defect2])


# The `defect_structure` for each complex is shown blow.
# 
# <img src="https://raw.githubusercontent.com/materialsproject/pymatgen-analysis-defects/main/docs/source/_static/img/defect_complex_equiv.png" width="800"/>
# 
# By inspection it is clear that `def_comp0` and `def_comp2` are symmetrically equivalent to each other and distinct from `def_comp1`, and our basic implementation of defect equivalence is able to verify this:
# 

# In[7]:


assert def_comp0 == def_comp2
assert def_comp0 != def_comp1


# However some defect complexes might become nonequivalent based on the periodic image you consider for the combination of sites.
# 
# <div class="alert alert-block alert-info"><b>Note:</b> To deal with these edge cases, we might have to add a dummy "DefectComplex" species at the "center" of the defect complex which will fix the selection of periodic for the different sites.  This is easy to implement but should be done when there is a good test case.
# </div>
# 

# ## Obtaining the simulation supercell
# 
# The simplest way to generate a supercell for simulating the defect is to just call the `get_supercell_structure` method for the defect.
# 
# Note that, under the hood, the `get_supercell_structure` method first attempts a "quick and dirty" inversion of a large cube in the bases of the lattice parameters.
# If a valid supercell is not discovered this way, ASE's `find_optimal_cell_shape` will be called to exhaustively search for the best supercell structure.
# 

# In[8]:


sc_struct = mg_ga_defect0.get_supercell_structure()
sc_struct.num_sites


# The supercell generated with default settings looks like this:
# 
# <img src="https://raw.githubusercontent.com/materialsproject/pymatgen-analysis-defects/main/docs/source/_static/img/defect_mg_ga_sc.png" width="300"/>
# 
# You can also reduce the `max_atoms` field to obtain a smaller simulation cell.
# Note that in this case, the conditions for the for cubic cell cannot be satisfied by the lattice vector inversion approach and the more expensive algorithm from ASE will be used.  Uncomment the following cell to see this in action.

# In[9]:


#sc_struct_smaller = mg_ga_defect0.get_supercell_structure(max_atoms=100)
#sc_struct_smaller.num_sites

