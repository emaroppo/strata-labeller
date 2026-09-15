"""Active learning over a catalog, and the Label Studio adapter that feeds it.

**May import:** ``labels``, ``catalog``, ``modelling`` and ``project``;
Label Studio only inside :mod:`strata.labeller.labelstudio`
(``docs/adr/0013``).

What another package may use is what is exported here; the job every tool
reads is ``strata.project``'s. See ``docs/adr/0015`` and ``docs/adr/0016``.
"""

from .config import Settings
from .project import LabellingProject, ProjectError

__all__ = ["LabellingProject", "ProjectError", "Settings"]
