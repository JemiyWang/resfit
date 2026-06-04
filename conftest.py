import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "manual: tests that require heavier dependencies (e.g. VitEncoder on real images) "
        "and are excluded from default CI runs (-m 'not manual').",
    )
