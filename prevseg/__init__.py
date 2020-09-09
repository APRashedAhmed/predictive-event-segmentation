import pkgutil
from prevseg._version import get_versions

__version__ = get_versions()['version']
__all__ = []

for loader, module_name, _ in  pkgutil.iter_modules(__path__):
    # Add the module name to all
    __all__.append(module_name)
    # Grab the module itself
    module = loader.find_module(module_name).load_module(module_name)
    # Add it to globals
    globals()[module_name] = module

# Cleanup the namespace
del get_versions
del pkgutil, loader, module_name, module, _
    
