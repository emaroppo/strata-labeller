"""Machine-level settings: how this host reaches Label Studio, its catalog and the modelling host.

Everything that belongs to a labelling job lives in the project directory
instead — see :mod:`strata.labeller.project`.

What a catalog is, and which one this host uses, is read through
:mod:`strata.catalog.config`, which the blob server and the modelling host
read too. This file owns only what is the labeller's own.
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from strata.catalog.config import Catalogs, read_catalogs


@dataclass
class LabelStudioConfig:
    url: str = "http://localhost:8080"
    api_key: str = ""
    # Where the images mount shows up inside the Label Studio container
    # Names a directory inside the Label Studio container, not a media:
    # whatever a project labels is served from it. The word stays because
    # deployments already mount it under this path — changing the default
    # would be a redeploy dressed up as a rename.
    local_storage_path: str = "/label-studio/data/images"
    #: What the catalog's blob directory is called inside the Label Studio
    #: container, for when Label Studio reads files off a mount rather than
    #: from a blob server. Must match the bind mount in docker-compose.yml.
    #: Here rather than with the catalog because it describes this
    #: container, and nothing but the labeller reads it.
    blobs_prefix: str = "blobs"


@dataclass
class ModellingConfig:
    """Where training happens.

    Empty means in this process, which is what a single machine wants and
    what keeps a checkout runnable. A URL sends rounds to a host with the
    GPU — it materialises the dataset itself, so nothing but a dataset id
    travels.
    """

    url: str = ""
    #: Shared with the host. Out of the file by preference, the same
    #: argument as every other credential here.
    token: str = ""


@dataclass
class Settings:
    label_studio: LabelStudioConfig = field(default_factory=LabelStudioConfig)
    #: The catalogs this host describes, and which one it uses by default.
    catalogs: Catalogs = field(default_factory=Catalogs)
    modelling: ModellingConfig = field(default_factory=ModellingConfig)

    @classmethod
    def load(cls, path: Path = Path("config.toml")) -> "Settings":
        data: dict = {}
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)

        settings = cls()
        if "label_studio" in data:
            settings.label_studio = LabelStudioConfig(**data["label_studio"])
        settings.catalogs = read_catalogs(data.get("catalog", {}))
        if "modelling" in data:
            settings.modelling = ModellingConfig(**data["modelling"])

        # Credentials belong in the environment rather than in a file that
        # gets copied around
        api_key = os.environ.get("LABEL_STUDIO_API_KEY")
        if api_key:
            settings.label_studio.api_key = api_key
        for name in ("url", "token"):
            value = os.environ.get(f"STRATA_MODELLING_{name.upper()}")
            if value:
                setattr(settings.modelling, name, value)

        return settings
