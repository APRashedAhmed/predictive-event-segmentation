"""Script for tests for all models in prevseg.models"""
import logging
from types import ModuleType

import pytest
import pytorch_lightning as pl

import prevseg
import prevseg.models as models
import prevseg.utils as utils

logger = logging.getLogger(__name__)

all_model_modules = utils.instances_and_names_in_module(prevseg.models,
                                                        ModuleType)
all_models = {name : utils.subclasses_and_names_in_module(
    mod, pl.LightningModule) for name, mod in all_model_modules}
name_model_class = [(module_name, cls)
                    for module_name, model_list in all_models.items()
                    for _, cls in model_list]

@pytest.mark.parametrize("name, cls", name_model_class)
def test_all_models(name, cls):
    # Make sure they can instantiate
    assert cls()
