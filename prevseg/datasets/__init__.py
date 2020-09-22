"""
Import all model submodules at runtime. lifted from this stack overflow post:
https://stackoverflow.com/questions/3365740/how-to-import-all-submodules

Extended to make all Pytorch Lightning models importable as well.
"""
import pkgutil

try:
    from torch.utils.data import Dataset
    import prevseg.utils as utils
    from prevseg.datasets.datamodule import DataModuleConstructor
except ModuleNotFoundError:
    pass

__all__ = []

for loader, module_name, _ in  pkgutil.walk_packages(__path__):
    # Add the module name to all
    __all__.append(module_name)
    # Grab the module itself
    module = loader.find_module(module_name).load_module(module_name)
    # Add it to globals
    globals()[module_name] = module

    try:
        # Get all the pytorch lightning models in the module
        name_cls_list = utils.subclasses_and_names_in_module(
            module, Dataset)
        for name, cls in name_cls_list:
            __all__.append(name)
            globals()[name] = cls
    except NameError:
        pass

# Cleanup the namespace
del pkgutil, loader, module_name, module, _
try:
    del Dataset, utils, name_cls_list, name, cls
except NameError:
    pass
