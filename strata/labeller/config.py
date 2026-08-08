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
    #: The index. Empty means SQLite under ``root``, which is what keeps a
    #: checkout runnable with nothing installed. A Postgres URL points
    #: several machines at one index, which is what a corpus of millions of
    #: rows needs — the schema and the queries are the same either way.
    url: str = ""

    #: Where the blobs are. Empty means files under ``root``; an endpoint
    #: means tar shards in an S3-compatible bucket, which is what lets the
    #: machine that trains and the machine that labels read the same bytes.
    s3_endpoint: str = ""
    s3_bucket: str = "strata"
    #: Garage and MinIO ignore it, but boto3 insists on one being set.
    s3_region: str = "garage"
    #: Kept out of the file by preference — $STRATA_S3_ACCESS_KEY and
    #: $STRATA_S3_SECRET_KEY override, the same argument as the LS token.
    s3_access_key: str = ""
    s3_secret_key: str = ""

    #: The sample-serving API, e.g. ``http://minipc:8081``. Empty means Label
    #: Studio reads images off the local blob mount, which is what every
    #: task created before the API does.
    serve_url: str = ""
    #: Signs blob URLs, and must match what the server was started with. An
    #: image tag cannot carry a header, so the URL is the credential — which
    #: is exactly why this belongs in $STRATA_BLOB_SECRET and not in a file.
    blob_secret: str = ""


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

        # Credentials belong in the environment rather than in a file that
        # gets copied around, the same argument as the Label Studio token
        url = os.environ.get("STRATA_CATALOG_URL")
        if url:
            settings.catalog.url = url
        secret = os.environ.get("STRATA_BLOB_SECRET")
        if secret:
            settings.catalog.blob_secret = secret
        for name in ("s3_endpoint", "s3_bucket", "s3_access_key", "s3_secret_key"):
            value = os.environ.get(f"STRATA_{name.upper()}")
            if value:
                setattr(settings.catalog, name, value)

        return settings
