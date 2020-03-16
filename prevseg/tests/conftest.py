"""Configuration for tests"""
import logging

import pytest

logger = logging.getLogger(__name__)

def pytest_addoption(parser):
    parser.addoption('--with_data',
                     action='store_true',
                     default=False,
                     help='Indicates data dirs are present and data-requiring '
                     'tests should be run',
    )

@pytest.fixture()
def with_data(request):
    return request.config.getoption('--with_data')
