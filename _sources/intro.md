# `pymatgen-analysis-defects`

This package is an extension to `pymatgen` for performing defect analysis.
The package is meant to be installed as a namespace package extension to the main `pymatgen` library.
The code will be accessible as `pymatgen.analysis.defects` after installation.
The new module has been redesigned to work closely with `atomate2`.

## Installation

The new `pymatgen-analysis-defects` module can be installed using `pip`:
```
pip install pymatgen-analysis-defects
```

## What happened to the old module?

The old `pymatgen.analysis.defects` module has been deprecated and removed from the main `pymatgen` library.
If you need to use the old module, you can install `pymatgen` version `<=2022.7.8` which is the last version that contains the old module.
This was a difficult decision, but the having two separate modules with the same name would be confusing and difficult to maintain.
The old module was not being actively developed and was not compatible with the new `atomate2` framework.


## Table of Contents

```{tableofcontents}

```
