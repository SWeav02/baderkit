# -*- coding: utf-8 -*-
import logging
from importlib import metadata

from rich.logging import RichHandler

# Version and Logging
__version__ = metadata.version("baderkit")

# Configure our logger to output timestamps with logs
# Also changes the logging level to info
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    handlers=[
        RichHandler(
            show_time=False,       # <-- Removes the timestamp completely
            show_level=False,      # <-- Removes the [INFO] tag for ultimate simplicity
            show_path=False,       # <-- Keeps the right-hand column clean
            markup=True,
            rich_tracebacks=True,
        )
    ],
)

from .bader.bader import Bader
from .toolkit.grid import Grid
from .toolkit.structure import Structure

# hide other imports
__all__ = ["Bader", "Structure", "Grid"]


def __dir__():
    return __all__
