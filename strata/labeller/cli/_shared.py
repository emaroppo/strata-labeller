"""What every command starts from: the app, the console, the options, and the preambles."""

from contextlib import contextmanager
from pathlib import Path

import typer
from rich import get_console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ..config import Settings
from ..project import PROJECT_ENV_VAR, PROJECTS_DIR, LabellingProject, ProjectError

app = typer.Typer(name="strata-labeller")


#: The console rich itself hands out, not one of our own. docs/adr/0030
console = get_console()


ProjectOption = typer.Option(
    None,
    "--project",
    "-p",
    help=(
        f"Project name under {PROJECTS_DIR}/, or a path "
        f"(default: ${PROJECT_ENV_VAR}, else the only project found)"
    ),
)


def _config_path(value: Path | None) -> Path:
    """The host file: ``--config``, else ``$STRATA_CONFIG``, else ``./config.toml``."""
    from strata.project import settings_path

    from strata.catalog.config import CatalogConfigError

    with _exit_on(CatalogConfigError):
        return settings_path(value)


ConfigOption = typer.Option(
    None,
    "--config",
    callback=_config_path,
    show_default="$STRATA_CONFIG, else config.toml",
    help="Host settings: catalogs, the modelling host, Label Studio",
)


#: For the commands with no project to ask: the operator names the
#: catalog. docs/adr/0020
CatalogOption = typer.Option(
    "", "--catalog", help="Which catalog on this host (default: the host's own)"
)


def _error(message: str) -> None:
    """Print an error. Escaped, since messages carry TOML section names."""
    console.print(escape(message), style="red")


@contextmanager
def _exit_on(*errors: type[Exception]):
    """Report a refusal and stop. Anything not named is a crash, and stays one."""
    try:
        yield
    except errors as e:
        _error(str(e))
        raise typer.Exit(1) from None


def _progress(
    *, bar: bool = False, elapsed: bool = False, remaining: bool = False, transient: bool = False
) -> Progress:
    """A progress display in the house style: a spinner and a description,
    then only what the caller can actually measure."""
    columns = [SpinnerColumn(), TextColumn("[progress.description]{task.description}")]
    if bar:
        columns += [BarColumn(), MofNCompleteColumn()]
    if elapsed:
        columns.append(TimeElapsedColumn())
    if remaining:
        columns.append(TimeRemainingColumn())
    return Progress(*columns, console=console, transient=transient)


def _load_project(path: Path | None) -> LabellingProject:
    with _exit_on(ProjectError):
        return LabellingProject.load(path)


def _ls_client(settings: Settings, project: LabellingProject, config_path: Path):
    """Build a Label Studio client, failing early on a missing token."""
    from ..labelstudio.ls_client import LSClient

    if not settings.label_studio.api_key:
        _error(
            f"No Label Studio API key. Set it in {config_path} "
            "(see config.example.toml) or in $LABEL_STUDIO_API_KEY."
        )
        raise typer.Exit(1)
    return LSClient(settings, project)


def _addressing(settings, config):
    """How this host writes and reads task image URLs, for one catalog.

    One place, because the two directions have to agree
    (``docs/adr/0013``). ``config`` is the project's catalog, and there is
    no default (``docs/adr/0020``).
    """
    from strata.catalog import SigningError

    from ..labelstudio.adapter import Addressing

    if config is None:
        raise TypeError(
            "_addressing needs the project's catalog config; the host's default "
            "catalog is not a stand-in for it."
        )
    with _exit_on(SigningError):
        return Addressing(prefix=config.blobs_prefix, urls=config.signed_urls())


def _local_paths(root: Path, samples) -> list[Path]:
    """Files on this host for ``samples``, for a model that reads paths.

    Not asked of the blob backend, which gives no paths
    (``docs/adr/0002``), and resolved from the checksum rather than the
    sample's location (``docs/adr/0001``).
    """
    from strata.catalog import blob_path

    blobs = root / "blobs"
    paths = []
    for sample in samples:
        suffix = Path((sample.metadata or {}).get("source_path") or "").suffix.lower()
        path = blobs / blob_path(sample.checksum, suffix)
        if not path.exists():
            _error(
                f"Predicting needs the bytes as files, and {path} is not on "
                f"this host. Three ways forward: set [modelling] url so a host "
                f"that has them scores the pool, run this where the blobs are, "
                f"or push without --predictions. Fetching a whole review pool "
                f"out of the bucket to rank it here is not something to do by "
                f"accident."
            )
            raise typer.Exit(1)
        paths.append(path)
    return paths


def _catalog_for(settings, config_path: Path, create: bool = False, name: str = ""):
    """One of this host's catalogs, or an exit with something actionable.

    ``name`` is the one a project asks for; empty means the host's default.
    ``create`` for the commands that put data in. See ``docs/adr/0018``.
    """
    from strata.catalog.config import CatalogMissing, open_catalog

    config = _catalog_config(settings, name)
    try:
        return open_catalog(config, create=create), Path(config.root)
    except CatalogMissing:
        _error(
            f"No catalog at {config.root}. Run 'strata-labeller ingest' to make one, "
            f"or point [catalog] root in {config_path} at an existing one."
        )
        raise typer.Exit(1) from None


def _settings(config_path: Path | None = None):
    """Host settings, or an exit saying what the file gets wrong.

    ``None`` resolves as ``--config`` does. Several catalogs and no
    default is refused at load. See ``docs/adr/0020``.
    """
    from strata.catalog.config import CatalogConfigError

    with _exit_on(CatalogConfigError):
        return Settings.load(config_path)


def _task_map(project, ls_project_id: int, catalog):
    """The cached task map for this project, or an exit explaining itself.

    One place, so the refusal of another catalog's map reads the same
    everywhere and an adopted map is said out loud. See ``docs/adr/0008``.
    """
    from ..labelstudio.sync import TaskMapError, load_task_map, task_map_catalog

    with _exit_on(TaskMapError):
        mapping = load_task_map(project, ls_project_id, catalog.id)

    if mapping and task_map_catalog(project, ls_project_id) is None:
        console.print(
            f"[dim]Task map has no catalog recorded; adopting it into "
            f"{catalog.id}. If this project has ever been pointed at another "
            f"catalog, run 'push --rebuild-map' instead.[/dim]"
        )
    return mapping


def _catalog_config(settings, name: str = ""):
    """Look up a named catalog, or exit saying which names exist.

    Never falls back to the default: see ``docs/adr/0020``.
    """
    from strata.catalog.config import CatalogConfigError

    with _exit_on(CatalogConfigError):
        return settings.catalogs.named(name)


def _catalog_if_any(settings, name: str = ""):
    """The configured catalog, or None when this host has none yet.

    For the paths that decorate output with counts, where absence is not an
    error. Distinct from :func:`_catalog_for`, which exits, and never a
    shortcut to a local index. See ``docs/adr/0020``.
    """
    from strata.catalog.config import CatalogMissing, open_catalog

    try:
        return open_catalog(_catalog_config(settings, name))
    except CatalogMissing:
        return None


def _warn_on_composition_drift(project: LabellingProject, catalog, schema) -> None:
    """Say so when a project's declarations do not match what it draws from.

    Warnings rather than refusals: afterwards the samples are the truth.
    See ``docs/adr/0010``.
    """
    declared_media = schema.media.name
    declared_subtype = type(project.sample_type()).subtype()
    held = catalog.samples.composition(project.collections)
    where = ", ".join(project.collections)

    other_media = {m: n for (m, _), n in held.items() if m != declared_media}
    if other_media:
        summary = ", ".join(f"{n:,} {m}" for m, n in sorted(other_media.items()))
        console.print(
            f"[yellow]This project labels {declared_media}, but {where} also "
            f"holds {summary}. Label Studio renders every task with the "
            f"{declared_media} tag, so those will not display.[/yellow]"
        )

    other_subtype = {
        sub: n for (m, sub), n in held.items() if m == declared_media and sub != declared_subtype
    }
    if other_subtype:
        summary = ", ".join(f"{n:,} {sub}" for sub, n in sorted(other_subtype.items()))
        console.print(
            f"[yellow]This project ingests as '{declared_subtype}', but {where} "
            f"holds {summary}. Grouping is recorded per sample, so what is "
            f"already there keeps its own — but anything ingested from here on "
            f"is grouped the way this project declares.[/yellow]"
        )


def _schema_for(project: LabellingProject, catalog):
    """The project's schema, with the classes the label set actually holds.

    Read from the catalog rather than from project.toml
    (``docs/adr/0014``). Falls back to the project's own when no label set
    exists yet, which is the case that creates one.
    """
    from strata.catalog import CatalogError

    try:
        _, label_set = catalog.label_sets.get(project.label_set_name)
    except CatalogError:
        return project.schema
    schema = project.schema_with(label_set.classes)
    _warn_on_composition_drift(project, catalog, schema)
    return schema


def _label_set_for(catalog, project: LabellingProject):
    from strata.catalog import CatalogError

    try:
        return catalog.label_sets.get(project.label_set_name)
    except CatalogError:
        _error(
            f"No label set named '{project.label_set_name}' in the catalog. "
            f"Run 'strata-labeller ingest', or set [catalog] label_set."
        )
        raise typer.Exit(1) from None
