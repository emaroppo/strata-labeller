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
class Settings:
    label_studio: LabelStudioConfig = field(default_factory=LabelStudioConfig)

    @classmethod
    def load(cls, path: Path = Path("config.toml")) -> "Settings":
        settings = cls()
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
            if "label_studio" in data:
                settings.label_studio = LabelStudioConfig(**data["label_studio"])

        api_key = os.environ.get("LABEL_STUDIO_API_KEY")
        if api_key:
            settings.label_studio.api_key = api_key

        return settings
