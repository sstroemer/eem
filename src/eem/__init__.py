import logging

logging.basicConfig(level=logging.WARNING, format="%(name)s ~ %(levelname)-7s :: %(message)s")
logging.getLogger("eem").setLevel(logging.DEBUG)

from .task import Task

from .entsoepywrapper import EntsoePyWrapper
from . import analysis
