# -*- coding: utf-8 -*-
from importlib import import_module
from typing import TYPE_CHECKING

# Set up lazy import

__all__ = [
    "BasinOverlap",
    "ElfLabeler",
    "SpinElfLabeler",
    "ElfRadii",
    "Badelf",
    "SpinBadelf",
    ]

_lazy_classes = {
    "BasinOverlap" : "baderkit.core.elf_analysis.overlap.overlap",
    "ElfLabeler" : "baderkit.core.elf_analysis.elf_labeler.elf_labeler",
    "SpinElfLabeler" : "baderkit.core.elf_analysis.elf_labeler.elf_labeler_spin",
    "ElfRadii" : "baderkit.core.elf_analysis.elf_radii.elf_radii",
    "Badelf" : "baderkit.core.elf_analysis.badelf.badelf",
    "SpinBadelf" : "baderkit.core.elf_analysis.badelf.badelf_spin",
}

def __getattr__(name: str):
    # check for class
    result = _lazy_classes.get(name, None)
    if result is not None:
        module = import_module(result)
        value = getattr(module, name)
        globals()[name] = value  # cache it so future access is fast
        return value

    # We've failed to find an attribute matching this name
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return __all__
# def __dir__():
#     return sorted(set(globals()) | set(__all__))

# if TYPE_CHECKING:
#     from .overlap import BasinOverlap # isort:skip
#     from .elf_labeler import ElfLabeler, SpinElfLabeler  # isort:skip
#     from .elf_radii import ElfRadii # isort:skip
#     from .badelf import Badelf, SpinBadelf  # isort:skip