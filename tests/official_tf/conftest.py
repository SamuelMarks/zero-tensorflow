"""Pytest configuration for ported `official_tf` test suite."""

from __future__ import annotations

from collections.abc import Iterator

from tests.conftest import install_fake_tensorflow

import pytest

install_fake_tensorflow()


@pytest.fixture(autouse=True)
def switcheroo_config() -> Iterator[None]:
    """Provide autouse fixture for `official_tf` tests.

    Yields:
        None: Execution control to test.
    """
    yield
