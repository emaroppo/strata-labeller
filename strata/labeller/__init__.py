"""Active learning over a catalog, and the Label Studio adapter that feeds it.

May import ``contracts``, ``catalog``, ``modelling`` and ``project``; Label Studio only
inside :mod:`strata.labeller.labelstudio`. See ``docs/adr/0013``, ``0015`` and ``0016``.
"""

from .config import Settings
from .project import LabellingProject, ProjectError

__all__ = ["LabellingProject", "ProjectError", "Settings"]
