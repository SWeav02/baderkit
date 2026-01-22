# example_api.py
from typing import Callable
from inspect import getdoc

def merge_param_sections(child_doc: str, base_doc: str) -> str:
    """Simple merger for 'Parameters' (NumPy style) or 'Args:' (Google style).
    This is intentionally small — adapt if you have different docstring conventions.
    """
    child_doc = child_doc or ""
    base_doc = base_doc or ""

    # Detect NumPy-style "Parameters" section
    if "Parameters" in base_doc:
        base_params = base_doc.split("Parameters", 1)[1].strip()
        if "Parameters" in child_doc:
            # Append missing part to child's Parameters block
            return child_doc + "\n\n" + base_params
        else:
            # Add a new Parameters heading and the base params
            sep = "\n\n" if child_doc else ""
            return child_doc + sep + "Parameters\n----------\n" + base_params

    # Try Google-style "Args:" fallback
    if "Args:" in base_doc:
        base_params = base_doc.split("Args:", 1)[1].strip()
        if "Args:" in child_doc:
            return child_doc + "\n\n" + base_params
        else:
            sep = "\n\n" if child_doc else ""
            return child_doc + sep + "Args:\n" + base_params

    # If nothing recognized, just append entire base doc
    return (child_doc + "\n\n" + base_doc).strip()


def inherit_signature_and_doc(base_fn: Callable):
    """Decorator: sets __signature__ on the wrapped fn and merges docstrings"""
    def decorator(fn: Callable):
        base_doc = getdoc(base_fn) or ""
        child_doc = getdoc(fn) or ""
        fn.__doc__ = merge_param_sections(child_doc, base_doc)
        return fn
    return decorator
