"""Tests for prevseg.index.py"""
import inspect
import logging
from pathlib import Path

import pytest

from prevseg import index
from prevseg.utils import instances_and_names_in_module, isiterable

logger = logging.getLogger(__name__)

test_dir_paths_and_names =  instances_and_names_in_module(index, Path)

@pytest.mark.parametrize("name, path", test_dir_paths_and_names)
def test_importable_dirs_exist(path, name, with_data):
    skip_names = index._data_prefixes
    if all([without not in name for without in skip_names]):
        assert path.exists()
