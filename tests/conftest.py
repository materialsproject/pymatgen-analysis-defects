from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_dir():
    module_dir = Path(__file__).resolve().parent
    test_dir = module_dir / "test_files"
    return test_dir.resolve()
