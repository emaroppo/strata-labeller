"""Machine-level settings: how to reach Label Studio on this host.

Everything that belongs to a labelling job lives in the project directory
instead — see :mod:`strata.labeller.project`.
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LabelStudioConfig:
    url: str = "http://localhost:8080"
    api_key: str = ""
    # Where the images mount shows up inside the Label Studio container
    local_storage_path: str = "/label-studio/data/images"


@dataclass
class CatalogConfig:
    """Where the catalog lives on this host.

    Machine-level for the same reason the Label Studio URL is: it describes
    this setup, not the job. Which *label set* inside it a job annotates
    against is the job's business and lives in project.toml.
    """

    root: str = "catalog"
    #: What the blobs mount is called inside the Label Studio container.
    #: Machine-level because the catalog is shared across projects, unlike
    #: the per-project data root it replaces.
    blobs_prefix: str = "blobs"


@dataclass
class Settings:
    label_studio: LabelStudioConfig = field(default_factory=LabelStudioConfig)
    catalog: CatalogConfig = field(default_factory=CatalogConfig)

    @classmethod
    def load(cls, path: Path = Path("config.toml")) -> "Settings":
        settings = cls()
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
            if "label_studio" in data:
                settings.label_studio = LabelStudioConfig(**data["label_studio"])
            if "catalog" in data:
                settings.catalog = CatalogConfig(**data["catalog"])

        api_key = os.environ.get("LABEL_STUDIO_API_KEY")
        if api_key:
            settings.label_studio.api_key = api_key

        return settings
