"""Script for utility functions in predictive-event-segmentation"""
import argparse
import inspect
import logging
from collections.abc import Iterable

logger = logging.getLogger(__name__)        

def as_list(obj, length=None, tp=None, iter_to_list=True):
    """Force an argument to be a list, optionally of a given length, optionally
    with all elements cast to a given type if not None.

    Parameters
    ---------
    obj : Object
        The obj we want to convert to a list.

    length : int or None, optional
        Length of new list. Applies if the inputted obj is not an iterable and
        iter_to_list is false.

    tp : type, optional
        Type to cast the values inside the list as.

    iter_to_list : bool, optional
        Determines if we should cast an iterable (not str) obj as a list or to
        enclose it in one.

    Returns
    -------
    obj : list
        The object enclosed or cast as a list.
    """
    # If the obj is None, return empty list or fixed-length list of Nones
    if obj is None:
        if length is None:
            return []
        return [None] * length
    
    # If it is already a list do nothing
    elif isinstance(obj, list):
        pass

    # If it is an iterable (and not str), convert it to a list
    elif isiterable(obj) and iter_to_list:
        obj = list(obj)
        
    # Otherwise, just enclose in a list making it the inputted length
    else:
        try:
            obj = [obj] * length
        except TypeError:
            obj = [obj]
        
    # Cast to type; Let exceptions here bubble up to the top.
    if tp is not None:
        obj = [tp(o) for o in obj]
    return obj

def isiterable(obj):
    """Function that determines if an object is an iterable, not including 
    str.

    Parameters
    ----------
    obj : object
        Object to test if it is an iterable.

    Returns
    -------
    bool : bool
        True if the obj is an iterable, False if not.
    """
    if isinstance(obj, str):
        return False
    else:
        return isinstance(obj, Iterable)

def flatten(inp_iter):
    """Recursively iterate through values in nested iterables, and return a
    flattened list of the inputted iterable.

    Parameters
    ----------
    inp_iter : iterable
        The iterable to flatten.

    Returns
    -------
    value : object
    	The contents of the iterable as a flat list.

    """
    def inner(inp):
        for val in inp:
            if isiterable(val):
                for ival in inner(val):
                    yield ival
            else:
                yield val
    return list(inner(inp_iter))

def _attrs_and_names_in_module(mode, module, cls=None):
    """Performs a check on the objects in a module and returns the object and
    name if it passes the check.
    
    Parameters
    ----------
    mode : function
    	Comparison function (ex isinstance, issubclass, etc.)
    
    module : Module
    	Module name to be searched through.

    Returns
    -------
    instances : list of tuples
    	List if name, instance pairs
    """
    instances = []
    all_instances = inspect.getmembers(module)
    for name, obj in all_instances:
        if cls is not None:
            try:
                if not mode(obj, cls):
                    continue
            except TypeError:
                continue
        instances.append((name, obj))
    return instances

def instances_and_names_in_module(module, cls=None):
    """Returns all instances of the passed class and their names.
    
    Parameters
    ----------
    module : Module
    	Module name to be searched through.

    cls : Class
    	Class to check each object against.

    Returns
    -------
    instances : list of tuples
    	List of name, instance pairs
    """
    return _attrs_and_names_in_module(isinstance, module, cls=cls)

def subclasses_and_names_in_module(module, cls=None):
    """Returns all subclasses of the inputed class.
    
    Parameters
    ----------
    module : Module
    	Module name to be searched through.

    cls : Class
    	Class to check each object against

    Returns
    -------
    subclasses : list of tuples
    	List of name, subclasses pairs
    """
    return _attrs_and_names_in_module(issubclass, module, cls=cls)

def child_argparser(parents, add_help=False, conflict_handler='resolve', *args,
                    **kwargs):
    return argparse.ArgumentParser(
        parents=parents if isiterable(parents) else [parents],
        add_help=add_help,
        conflict_handler=conflict_handler,
        *args, **kwargs
    )
