# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the Modified BSD License.

from pathlib import Path

from setuptools import find_namespace_packages, setup

module_dir = Path(__file__).resolve().parent

with open(module_dir / "README.md") as f:
    long_description = f.read()
setup(
    name="pymatgen-analysis-defects",
    packages=find_namespace_packages(include=["pymatgen.analysis.*"]),
    version="0.0.2",
    install_requires=["pymatgen>=2022.2.3"],
    extras_require={},
    package_data={},
    author="Jimmy Shen",
    author_email="jmmshn@gmail.com",
    maintainer="Jimmy Shen",
    url="https://github.com/materialsproject/pymatgen-analysis-defects",
    license="BSD",
    description="Add-ons to pymatgen for defect analysis.",
    long_description=long_description,
    keywords=["pymatgen"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
