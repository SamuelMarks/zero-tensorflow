"""Pytest configuration for ported `official_tf` test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def switcheroo_config() -> Iterator[None]:
    """Provide autouse fixture for `official_tf` tests.

    Yields:
        None: Execution control to test.
    """
    yield
