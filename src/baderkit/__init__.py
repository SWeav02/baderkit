# -*- coding: utf-8 -*-
from importlib import metadata
import logging
from rich.logging import RichHandler

# Version and Logging
__version__ = metadata.version("baderkit")

# Configure our logger to output timestamps with logs
# Also changes the logging level to info
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        RichHandler(
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
    ],
)

from .toolkit.structure import Structure
from .toolkit.grid import Grid
from .bader.bader import Bader


# hide other imports
__all__ = ["Bader", "Structure", "Grid"]
def __dir__():
    return __all__