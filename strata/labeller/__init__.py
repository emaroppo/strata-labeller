"""Active learning over a catalog, and the Label Studio adapter that feeds it.

**May import:** ``labels``, ``catalog``, ``modelling`` and ``project``;
Label Studio only inside :mod:`strata.labeller.labelstudio`
(``docs/adr/0013``).

What another package may use is what is exported here (``docs/adr/0015``).
Nothing in the workspace needs anything of it today: the job every tool
reads is ``strata.project``'s, and this package is a consumer of it too.
"""

from .config import Settings
from .project import LabellingProject, ProjectError

__all__ = ["LabellingProject", "ProjectError", "Settings"]
