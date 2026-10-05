"""Machine-level settings: the host's, plus how this host reaches Label Studio.

The catalogs and the modelling host are :class:`strata.project.Settings`,
which every tool on the machine reads. This adds the one section that is
the labeller's own. Everything that belongs to a job lives in the project
directory instead; see :mod:`strata.labeller.project`.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

from strata.catalog.config import CatalogConfigError
from strata.project import ModellingConfig
from strata.project import Settings as HostSettings

__all__ = ["LabelStudioConfig", "LabelStudioConfigError", "ModellingConfig", "Settings"]


class LabelStudioConfigError(CatalogConfigError):
    """``[label_studio]`` describes no instance this command can use."""


@dataclass
class LabelStudioConfig:
    url: str = "http://localhost:8080"
    api_key: str = field(default="", repr=False)
    # A directory inside the Label Studio container, not a media; the word
    # stays because deployments mount it there. docs/adr/0013
    local_storage_path: str = "/label-studio/data/images"
    #: The catalog whose own instance this is; empty for the machine's.
    catalog: str = ""

    def key_from(self) -> str:
        """Where its API key is read from, for the error that says it is missing."""
        if not self.catalog:
            return "$LABEL_STUDIO_API_KEY"
        return f"${_key_variable(self.catalog)}"


#: What ``[label_studio]`` and each ``[label_studio.<catalog>]`` may say.
_KEYS = frozenset(f.name for f in fields(LabelStudioConfig)) - {"catalog"}


def _key_variable(catalog: str) -> str:
    """``main``'s instance reads ``$LABEL_STUDIO_API_KEY_MAIN``."""
    return "LABEL_STUDIO_API_KEY_" + re.sub(r"[^A-Za-z0-9]", "_", catalog).upper()


@dataclass
class Settings(HostSettings):
    #: The machine's Label Studio, for a catalog with none of its own.
    label_studio: LabelStudioConfig = field(default_factory=LabelStudioConfig)
    #: From ``[label_studio.<catalog>]``: a catalog labelled on an instance
    #: of its own. docs/adr/0044
    label_studio_by_catalog: dict[str, LabelStudioConfig] = field(default_factory=dict)

    def _read(self, data: dict, environ: Mapping[str, str]) -> None:
        super()._read(data, environ)
        section = data.get("label_studio", {})
        scalars = {k: v for k, v in section.items() if not isinstance(v, dict)}
        tables = {k: v for k, v in section.items() if isinstance(v, dict)}
        for where, keys in [("label_studio", scalars)] + [
            (f"label_studio.{n}", t) for n, t in tables.items()
        ]:
            unknown = set(keys) - _KEYS
            if unknown:
                raise LabelStudioConfigError(
                    f"Unknown key(s) in [{where}]: {', '.join(sorted(unknown))} "
                    f"(known: {', '.join(sorted(_KEYS))})"
                )

        self.label_studio = LabelStudioConfig(**scalars)
        # Credentials come from the environment. docs/adr/0019
        api_key = environ.get("LABEL_STUDIO_API_KEY")
        if api_key:
            self.label_studio.api_key = api_key

        known = self.catalogs.names()
        self.label_studio_by_catalog = {}
        for name, table in tables.items():
            if name not in known:
                raise LabelStudioConfigError(
                    f"[label_studio.{name}] names a catalog this machine does not "
                    f"describe. Configured: {', '.join(known)}."
                )
            config = LabelStudioConfig(**{**scalars, **table}, catalog=name)
            # Its own key only: the machine's belongs to another instance, and
            # sending it would be refused as an invalid token by the wrong server
            config.api_key = environ.get(_key_variable(name)) or table.get("api_key", "")
            self.label_studio_by_catalog[name] = config

    def label_studio_for(self, catalog: str = "") -> LabelStudioConfig:
        """The instance a project on ``catalog`` is labelled on.

        The catalog's own when it has one, else the machine's. An empty
        ``catalog`` is the machine's default catalog. See ``docs/adr/0044``.
        """
        name = catalog or self.catalogs.default_name
        return self.label_studio_by_catalog.get(name, self.label_studio)
