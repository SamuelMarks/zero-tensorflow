import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")
    ),
)
import pytest


@pytest.fixture(autouse=True)
def switcheroo_config():
    # Unified pytest configuration that imports switcheroo config contexts
    if True:
        yield
