"""
Import all model submodules at runtime. lifted from this stack overflow post:
https://stackoverflow.com/questions/3365740/how-to-import-all-submodules
"""
import pkgutil

__all__ = []
for loader, module_name, _ in  pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    globals()[module_name] = loader.find_module(module_name).load_module(
        module_name)
