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


#: The console rich itself hands out, not one of our own. Two Console
#: objects writing to one terminal cannot coordinate: a live display owned
#: by one knows nothing about text printed through the other, and the two
#: fight over the same lines — which is what made a progress bar flicker
#: against a model's own output.
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


ConfigOption = typer.Option(
    "config.toml", "--config", help="Host settings: Label Studio URL and API key"
)


#: For the commands with no project to ask. A project names its catalog in
#: project.toml; these operate on the host, so the operator names it.
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

    One place, because the two directions have to agree: pushing HTTP URLs
    while reading local ones would orphan every task, and the symptom is an
    empty export rather than an error.

    ``config`` is the project's catalog, and there is no default. Falling
    back to the host's default catalog was how two callers signed URLs with
    the wrong catalog's secret: right on every host with one catalog,
    wrong on a host with several, and nothing to say which.
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

    Deliberately not asked of the blob backend. Once blobs are shards in a
    bucket there is no path to give — a tar member is not a file — but the
    local copies stay on the machine that ingested them, and predicting over
    a review pool is exactly the work that happens there.

    Resolved from the checksum rather than from the sample's location, which
    is what makes this keep working after a repack points every row at a
    shard.
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
    ``create`` for the commands that put data in: refusing to make one would
    leave no way to make the first, and the advice would be circular.
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


def _settings(config_path: Path = Path("config.toml")):
    """Host settings, or an exit saying what the file gets wrong.

    Loading can now fail on its own: a host with several catalogs and
    nothing saying which is the default is refused rather than guessed at.
    A traceback would be a poor way to say so.
    """
    from strata.catalog.config import CatalogConfigError

    with _exit_on(CatalogConfigError):
        return Settings.load(config_path)


def _task_map(project, ls_project_id: int, catalog):
    """The cached task map for this project, or an exit explaining itself.

    One place, so the refusal of another catalog's map reads the same
    everywhere and an adopted map is said out loud.
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

    Never falls back to the default: see ``docs/adr/0008``.
    """
    from strata.catalog.config import CatalogConfigError

    with _exit_on(CatalogConfigError):
        return settings.catalogs.named(name)


def _catalog_if_any(settings, name: str = ""):
    """The configured catalog, or None when this host has none yet.

    For the paths that decorate output with counts, where absence is not an
    error: a project that has never ingested is still a project. Distinct
    from :func:`_catalog_for`, which exits — these callers must not.

    What it is not is a shortcut to the local one. Checking for a
    ``catalog.db`` and opening it regardless of configuration is how five
    commands ended up reading a stale SQLite index after the catalog moved
    to Postgres, reporting counts from it as though they were current.
    """
    from strata.catalog.config import CatalogMissing, open_catalog

    try:
        return open_catalog(_catalog_config(settings, name))
    except CatalogMissing:
        return None


def _warn_on_composition_drift(project: LabellingProject, catalog, schema) -> None:
    """Say so when a project's declarations do not match what it draws from.

    Both are declared because ``ingest`` needs them before anything is
    catalogued: media picks which files count, and kind decides whether they
    are grouped. Afterwards the samples are the truth.

    Warnings rather than refusals. A mixed collection is legitimate — one
    catalog holding standalone images and video frames at once was the point
    of putting grouping on the sample — so the useful thing is to say what is
    there, not to stop.
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
        sub: n for (m, sub), n in held.items()
        if m == declared_media and sub != declared_subtype
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

    Read from the catalog rather than from project.toml, so the list a
    reviewer is offered and the list an export is validated against cannot
    disagree. Falls back to the project's own when no label set exists yet —
    which is the case that creates one.
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
