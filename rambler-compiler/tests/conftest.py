"""Shared fixtures for the CPU-only test suite.

The data/, evaluation/, and config/ tiers must never import torch; training/
imports torch lazily inside functions. See tests/test_no_torch_imports.py.
"""

import random

import pytest


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)
