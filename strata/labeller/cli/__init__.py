"""The labeller's command line, one module per group of commands.

``app`` is what the console script runs. Importing the command modules is
what registers their commands on it, in the order they are listed.
"""

from . import data, hosts, projects, queue, training  # noqa: F401
from ._shared import app

__all__ = ["app"]
