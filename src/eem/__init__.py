import logging

logging.basicConfig(level=logging.WARNING, format="%(name)s ~ %(levelname)-7s :: %(message)s")
if "eem" not in logging.Logger.manager.loggerDict.keys():
    logging.getLogger("eem").setLevel(logging.INFO)

from .task import Task
from .entsoepywrapper import EntsoePyWrapper

from . import analysis
from . import llm
