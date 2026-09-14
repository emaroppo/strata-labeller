"""Active learning over a catalog, and the Label Studio adapter that feeds it.

**May import:** ``labels``, ``catalog`` and ``modelling``; Label Studio only
inside :mod:`strata.labeller.labelstudio` (``docs/adr/0013``).

What another package may use is what is exported here (``docs/adr/0015``):
the project construct, which names what a job is made of, and the
machine's settings. Everything else in this package is the labelling loop's
own.
"""

from .config import Settings
from .project import Project, ProjectError

__all__ = ["Project", "ProjectError", "Settings"]
