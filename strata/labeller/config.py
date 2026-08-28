"""Machine-level settings: how to reach Label Studio on this host.

Everything that belongs to a labelling job lives in the project directory
instead — see :mod:`strata.labeller.project`.
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

#: The name a host's single catalog goes by when the file names none.
DEFAULT_CATALOG = "default"


class ConfigError(Exception):
    """config.toml says something this cannot act on."""


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
    #: The catalog used when a job names none. A host with one catalog has
    #: only this, which is why every caller that does not care can keep
    #: reading it.
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    #: Catalogs by name, from ``[catalog.<name>]`` tables. Empty on a host
    #: with a single flat ``[catalog]``.
    catalogs: dict[str, CatalogConfig] = field(default_factory=dict)
    #: Which name :attr:`catalog` resolved to. Empty when several are
    #: configured and none is marked default — see :meth:`catalog_named`.
    default_catalog: str = DEFAULT_CATALOG
    modelling: ModellingConfig = field(default_factory=ModellingConfig)

    def catalog_names(self) -> list[str]:
        return sorted(self.catalogs) or [DEFAULT_CATALOG]

    def catalog_named(self, name: str = "") -> CatalogConfig:
        """One catalog by name, or the default when nothing is named.

        A name that does not exist is refused with the list of the ones
        that do. Falling back to the default is the failure this naming
        exists to prevent: sample ids mean nothing outside the catalog that
        issued them, so a job reading the wrong one reports real numbers
        about the wrong data without anything raising.

        Asking for "the default" where several exist and none is marked is
        refused for the same reason — but only here, at the point of the
        ambiguity. A project or ``--catalog`` naming one settles it, and
        loading the file cannot know whether anyone will.
        """
        if not name:
            if not self.default_catalog and self.catalogs:
                raise ConfigError(
                    f"This host has several catalogs "
                    f"({', '.join(sorted(self.catalogs))}) and nothing says "
                    f"which is the default. Add [catalog] default = \"<name>\", "
                    f"name one in a project's [catalog] name, or pass --catalog."
                )
            return self.catalog
        try:
            return self.catalogs[name]
        except KeyError:
            known = ", ".join(self.catalog_names())
            raise ConfigError(
                f"No catalog named {name!r} on this host. Configured: {known}."
            ) from None

    @classmethod
    def load(cls, path: Path = Path("config.toml")) -> "Settings":
        settings = cls()
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
            if "label_studio" in data:
                settings.label_studio = LabelStudioConfig(**data["label_studio"])
            if "catalog" in data:
                settings._read_catalogs(data["catalog"])
            if "modelling" in data:
                settings.modelling = ModellingConfig(**data["modelling"])

        api_key = os.environ.get("LABEL_STUDIO_API_KEY")
        if api_key:
            settings.label_studio.api_key = api_key

        # Credentials belong in the environment rather than in a file that
        # gets copied around, the same argument as the Label Studio token
        # Names one index, so it lands on the default catalog rather than
        # on all of them: pointing every catalog at one database would make
        # the host claim two corpora where it has one, and identity — which
        # merging depends on — would agree with the lie.
        url = os.environ.get("STRATA_CATALOG_URL")
        if url:
            settings.catalog.url = url
        for name in ("url", "token"):
            value = os.environ.get(f"STRATA_MODELLING_{name.upper()}")
            if value:
                setattr(settings.modelling, name, value)

        # These describe the host's storage and its credentials rather than
        # any one catalog, so they reach every catalog configured on it
        secret = os.environ.get("STRATA_BLOB_SECRET")
        for catalog in settings._every_catalog():
            if secret:
                catalog.blob_secret = secret
            for name in ("s3_endpoint", "s3_bucket", "s3_access_key", "s3_secret_key"):
                value = os.environ.get(f"STRATA_{name.upper()}")
                if value:
                    setattr(catalog, name, value)

        return settings

    def _every_catalog(self):
        seen = {id(self.catalog)}
        yield self.catalog
        for catalog in self.catalogs.values():
            if id(catalog) not in seen:
                yield catalog

    def _read_catalogs(self, section: dict) -> None:
        """A flat ``[catalog]``, or ``[catalog.<name>]`` tables, or both.

        Scalar keys are the host's own settings — where the blobs are, which
        endpoint, which secret — and named tables layer on top of them, so
        two catalogs in one bucket state the bucket once. A file with no
        named tables is the single-catalog case and behaves exactly as it
        did before.
        """
        scalars = {k: v for k, v in section.items() if not isinstance(v, dict)}
        tables = {k: v for k, v in section.items() if isinstance(v, dict)}
        chosen = scalars.pop("default", "")

        base = _catalog_config(scalars, "catalog")
        self.catalogs = {
            name: _catalog_config({**scalars, **table}, f"catalog.{name}")
            for name, table in tables.items()
        }

        if not self.catalogs:
            if chosen:
                raise ConfigError(
                    f"[catalog] default = {chosen!r}, but no [catalog.<name>] "
                    f"tables are defined. This host has one catalog."
                )
            self.catalog, self.default_catalog = base, DEFAULT_CATALOG
            return
        if chosen:
            if chosen not in self.catalogs:
                raise ConfigError(
                    f"[catalog] default = {chosen!r} names no catalog. "
                    f"Configured: {', '.join(sorted(self.catalogs))}."
                )
            self.default_catalog = chosen
        elif len(self.catalogs) == 1:
            self.default_catalog = next(iter(self.catalogs))
        else:
            # Left unresolved rather than guessed, and not an error yet: a
            # project or --catalog naming one settles it. Picking one
            # alphabetically would hand over a corpus chosen by spelling.
            self.default_catalog = ""
            self.catalog = base
            return
        self.catalog = self.catalogs[self.default_catalog]


def _catalog_config(values: dict, where: str) -> CatalogConfig:
    try:
        return CatalogConfig(**values)
    except TypeError as e:
        raise ConfigError(f"[{where}]: {e}") from None
