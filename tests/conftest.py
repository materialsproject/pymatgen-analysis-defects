from pathlib import Path

import pytest
from pymatgen.core import Structure


@pytest.fixture(scope="session")
def test_dir():
    module_dir = Path(__file__).resolve().parent
    test_dir = module_dir / "test_files"
    return test_dir.resolve()


@pytest.fixture(scope="session")
def gan_struct(test_dir):
    return Structure.from_file(test_dir / "GaN.vasp")
