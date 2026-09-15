"""Machine-level settings: the host's, plus how this host reaches Label Studio.

The catalogs and the modelling host are :class:`strata.project.Settings`,
which every tool on the machine reads. This adds the one section that is
the labeller's own. Everything that belongs to a job lives in the project
directory instead; see :mod:`strata.labeller.project`.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from strata.project import ModellingConfig
from strata.project import Settings as HostSettings

__all__ = ["LabelStudioConfig", "ModellingConfig", "Settings"]


@dataclass
class LabelStudioConfig:
    url: str = "http://localhost:8080"
    api_key: str = ""
    # A directory inside the Label Studio container, not a media; the word
    # stays because deployments mount it there. docs/adr/0013
    local_storage_path: str = "/label-studio/data/images"


@dataclass
class Settings(HostSettings):
    label_studio: LabelStudioConfig = field(default_factory=LabelStudioConfig)

    def _read(self, data: dict, environ: Mapping[str, str]) -> None:
        super()._read(data, environ)
        if "label_studio" in data:
            self.label_studio = LabelStudioConfig(**data["label_studio"])
        # Credentials come from the environment. docs/adr/0019
        api_key = environ.get("LABEL_STUDIO_API_KEY")
        if api_key:
            self.label_studio.api_key = api_key
