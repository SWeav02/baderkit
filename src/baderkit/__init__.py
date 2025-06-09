# -*- coding: utf-8 -*-
# high level imports
import importlib.metadata
import logging

from rich.logging import RichHandler

from .core import Bader

__version__ = importlib.metadata.version("baderkit")

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
