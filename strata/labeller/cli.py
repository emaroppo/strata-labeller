import json
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
from rich.table import Table

from strata.labels import feature_digest

from .config import Settings
from .project import PROJECT_ENV_VAR, PROJECTS_DIR, Project, ProjectError

app = typer.Typer(name="auto-labeller")
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


def _load_project(path: Path | None) -> Project:
    try:
        return Project.load(path)
    except ProjectError as e:
        _error(str(e))
        raise typer.Exit(1) from None


def _ls_client(settings: Settings, project: Project, config_path: Path):
    """Build a Label Studio client, failing early on a missing token."""
    from .ls_client import LSClient

    if not settings.label_studio.api_key:
        _error(
            f"No Label Studio API key. Set it in {config_path} "
            "(see config.example.toml) or in $LABEL_STUDIO_API_KEY."
        )
        raise typer.Exit(1)
    return LSClient(settings, project)


def _addressing(settings, config=None):
    """How this host writes and reads task image URLs.

    One place, because the two directions have to agree: pushing HTTP URLs
    while reading local ones would orphan every task, and the symptom is an
    empty export rather than an error.
    """
    from .adapter import AdapterError, Addressing

    config = config or settings.catalogs.default
    try:
        return Addressing(
            prefix=settings.label_studio.blobs_prefix,
            base_url=config.serve_url,
            secret=config.blob_secret,
        )
    except AdapterError as e:
        _error(str(e))
        raise typer.Exit(1) from None


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


def _redacted(url: str) -> str:
    """An index URL safe to print.

    Every command that reports where the catalog is gets run when something
    is broken, and its output gets pasted into a chat window or an issue.
    A connection URL carries its password inline, so printing it raw makes
    routine troubleshooting leak a credential.
    """
    from sqlalchemy.engine import make_url

    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        # Not a URL SQLAlchemy recognises. Saying so beats printing it.
        return "<unparseable url>"


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
            f"No catalog at {config.root}. Run 'auto-labeller ingest' to make one, "
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

    try:
        return Settings.load(config_path)
    except CatalogConfigError as e:
        _error(str(e))
        raise typer.Exit(1) from None


def _task_map(project, ls_project_id: int, catalog):
    """The cached task map for this project, or an exit explaining itself.

    Loaded through here rather than directly so that the refusal — a map
    written against another catalog — reads the same wherever it happens,
    and so that adopting a map from before identities were recorded is
    said out loud rather than assumed.
    """
    from .sync import TaskMapError, load_task_map, task_map_catalog

    try:
        mapping = load_task_map(project, ls_project_id, catalog.id)
    except TaskMapError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    if mapping and task_map_catalog(project, ls_project_id) is None:
        console.print(
            f"[dim]Task map has no catalog recorded; adopting it into "
            f"{catalog.id}. If this project has ever been pointed at another "
            f"catalog, run 'push --rebuild-map' instead.[/dim]"
        )
    return mapping


def _catalog_config(settings, name: str = ""):
    """Look up a named catalog, or exit saying which names exist.

    A name that resolves to nothing must not fall back to the default. The
    ids in a catalog mean nothing outside it, so a job quietly reading the
    wrong one is the failure this naming exists to prevent.
    """
    from strata.catalog.config import CatalogConfigError

    try:
        return settings.catalogs.named(name)
    except CatalogConfigError as e:
        _error(str(e))
        raise typer.Exit(1) from None


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


def _warn_on_composition_drift(project: Project, catalog, schema) -> None:
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
    held = catalog.composition(project.collections)
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


def _schema_for(project: Project, catalog):
    """The project's schema, with the classes the label set actually holds.

    Read from the catalog rather than from project.toml, so the list a
    reviewer is offered and the list an export is validated against cannot
    disagree. Falls back to the project's own when no label set exists yet —
    which is the case that creates one.
    """
    from strata.catalog import CatalogError

    try:
        _, label_set = catalog.label_set(project.label_set_name)
    except CatalogError:
        return project.schema
    schema = project.schema_with(label_set.classes)
    _warn_on_composition_drift(project, catalog, schema)
    return schema


def _label_set_for(catalog, project: Project):
    from strata.catalog import CatalogError

    try:
        return catalog.label_set(project.label_set_name)
    except CatalogError:
        _error(
            f"No label set named '{project.label_set_name}' in the catalog. "
            f"Run 'auto-labeller ingest', or set [catalog] label_set."
        )
        raise typer.Exit(1) from None


@app.command()
def new(
    name_or_path: str = typer.Argument(
        ..., metavar="NAME", help=f"Project name (created under {PROJECTS_DIR}/) or a path"
    ),
    name: str | None = typer.Option(None, help="Project name (default: directory name)"),
    classes: list[str] = typer.Option([], "--class", help="Label class (repeatable)"),
    single: bool = typer.Option(False, "--single", help="Classes are mutually exclusive"),
    template: str = typer.Option(
        "image_classification", help="Label config template (see 'templates')"
    ),
) -> None:
    """Scaffold a new project under projects/ (or at an explicit path)."""
    directory = Path(name_or_path)
    # A bare name lands in projects/; anything path-shaped is taken literally
    if len(directory.parts) == 1 and not directory.is_absolute():
        directory = Path(PROJECTS_DIR) / directory
    directory.mkdir(parents=True, exist_ok=True)
    try:
        project = Project.create(
            directory,
            name=name,
            classes=list(classes),
            choice="single" if single else "multiple",
            template=template,
        )
    except ProjectError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    console.print(f"[green]Created project '{project.name}' in {directory}[/green]")
    console.print(f"  Put {project.schema.media.name} files in {project.data_dir}, then run:")
    console.print(f"    auto-labeller ingest --project {project.name}")


@app.command()
def templates() -> None:
    """List the available label config templates."""
    from . import schemas

    table = Table(title="Label config templates")
    table.add_column("Template", style="cyan")
    table.add_column("Media", style="magenta")
    table.add_column("Annotations", style="green")
    for name in schemas.available_templates():
        if name == schemas.CUSTOM_TEMPLATE:
            table.add_row(name, "-", "whatever label_config.xml declares")
            continue
        spec = schemas.TEMPLATES[name]
        table.add_row(name, spec.media.name, spec.control_tag)
    console.print(table)
    # Which files count is the catalog's answer, not a template's
    console.print("[dim]Which files a project ingests comes from its "
                  "[data] type — see 'auto-labeller types'.[/dim]")


@app.command(name="catalogs")
def catalogs_cmd(config_path: Path = ConfigOption) -> None:
    """List the catalogs this host is configured for.

    A host with one has one, called 'default', whether or not the file says
    so. Which one a project draws from is in its own [catalog] name.
    """
    settings = _settings(config_path)

    table = Table(title="Catalogs")
    table.add_column("Name", style="cyan")
    table.add_column("Index", style="magenta")
    table.add_column("Blobs", style="green")
    table.add_column("Identity", style="yellow")
    for name in settings.catalogs.names():
        config = settings.catalogs.named("" if name == settings.catalogs.default_name else name)
        index = _redacted(config.url) if config.url else f"sqlite under {config.root}"
        blobs = (
            f"{config.s3_endpoint}/{config.s3_bucket}"
            if config.s3_endpoint
            else f"files under {config.root}/blobs"
        )
        # Asked of the catalog rather than read from the file: two names
        # pointing at one database is the mistake this makes visible, and
        # only the catalog itself can say.
        try:
            catalog = _catalog_if_any(
                settings, "" if name == settings.catalogs.default_name else name
            )
            identity = catalog.id if catalog is not None else "[dim]not created yet[/dim]"
        except Exception as e:  # a catalog that cannot be reached is not fatal here
            identity = f"[red]{escape(str(e).splitlines()[0])}[/red]"
        label = (
            f"{name} [dim](default)[/dim]" if name == settings.catalogs.default_name else name
        )
        table.add_row(label, index, blobs, identity)
    console.print(table)


@app.command(name="types")
def sample_types() -> None:
    """List the sample types this catalog can ingest."""
    from strata.catalog.sample_types import SampleType, available, resolve

    table = Table(title="Sample types")
    table.add_column("Type", style="cyan")
    table.add_column("Media", style="magenta")
    table.add_column("Subtype", style="green")
    table.add_column("Files", style="yellow")
    table.add_column("Groups", style="blue")
    for name in sorted(available()):
        cls = resolve(name)
        table.add_row(
            name,
            cls.media,
            cls.subtype(),
            ", ".join(f".{e}" for e in sorted(cls.extensions)) or "-",
            # Whether it overrides grouping, which is the difference between
            # frames staying together and each one being its own group
            "yes" if cls.group_id_for is not SampleType.group_id_for else "-",
        )
    console.print(table)
    console.print(
        "[dim]Read from what is installed. A plugin registers here too, and "
        "cannot replace a built-in name.[/dim]"
    )


@app.command(name="preparers")
def list_preparers() -> None:
    """List the conversions installed here, and what each turns into what."""
    from strata.catalog.preparers import available, resolve

    table = Table(title="Preparers")
    table.add_column("Name", style="cyan")
    table.add_column("Reads", style="yellow")
    table.add_column("Produces", style="green")
    for name in sorted(available()):
        cls = resolve(name)
        table.add_row(
            name,
            ", ".join(f".{e}" for e in sorted(cls.sources)) or "-",
            cls.produces,
        )
    if not available():
        console.print(
            "[yellow]None installed. A conversion is a plugin — it carries a "
            "decoder or a parser, and a checkout with nothing to convert "
            "should not have to install one.[/yellow]"
        )
        return
    console.print(table)
    console.print(
        r"[dim]'prepare' writes into \[data] root, then 'ingest' catalogues "
        "it. Two steps, because ingest is where content addressing, "
        "grouping and collections are decided.[/dim]"
    )


@app.command(name="projects")
def list_projects_cmd() -> None:
    """List the projects under projects/."""
    from strata.catalog import CatalogError

    from .project import list_projects

    found = list_projects()
    if not found:
        console.print(
            "[yellow]No projects yet. Create one with 'auto-labeller new <name>'.[/yellow]"
        )
        return

    # Counts live in the catalog now, and a project can be listed without
    # one — a project that has never ingested is still a project.
    #
    # Opened per project rather than once: two projects on one host need
    # not draw from the same catalog, and counting both against whichever
    # happens to be the default is how a listing reports numbers that
    # belong to another corpus.
    settings = _settings()
    catalogs: dict[str, object] = {}

    def catalog_for_project(project):
        name = project.catalog.name
        if name not in catalogs:
            catalogs[name] = _catalog_if_any(settings, name)
        return catalogs[name]

    table = Table(title="Projects")
    table.add_column("Name", style="cyan")
    table.add_column("Schema", style="magenta")
    table.add_column("Classes", style="green")
    table.add_column("Samples", justify="right")
    table.add_column("Labeled", justify="right")
    table.add_column("LS id", justify="right")

    for directory in found:
        try:
            project = Project.load(directory)
        except ProjectError as e:
            table.add_row(directory.name, "-", f"[red]{escape(str(e))}[/red]", "-", "-", "-")
            continue
        total, labeled = "-", "-"
        catalog = catalog_for_project(project)
        if catalog is not None:
            try:
                label_set_id, _ = catalog.label_set(project.label_set_name)
            except CatalogError:
                pass
            else:
                where = project.collections
                annotated = catalog.labelled(label_set_id, where)
                queued = catalog.unlabelled(label_set_id, where)
                skipped = catalog.skipped(label_set_id, where)
                total = str(len(annotated) + len(queued) + len(skipped))
                labeled = str(len(annotated))
        table.add_row(
            project.name,
            project.schema.type,
            ", ".join(project.schema.classes) or "-",
            total,
            labeled,
            str(project.label_studio.project_id or "-"),
        )
    console.print(table)


class_app = typer.Typer(name="class", help="Inspect and extend a project's label classes.")
app.add_typer(class_app, name="class")


@class_app.command("add")
def class_add(
    names: list[str] = typer.Argument(..., help="Class name(s) to add"),
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
    push: bool = typer.Option(
        True, "--push/--no-push", help="Also add the class to the Label Studio config"
    ),
) -> None:
    """Add a class to the project, and to Label Studio's labeling config.

    The live config is edited in place rather than regenerated, so a
    hand-tuned layout survives. Refresh your Label Studio tab afterwards and
    the new option is there.
    """
    from .label_config import LabelConfigError

    project = _load_project(project_path)
    settings = _settings(config_path)

    try:
        # No inference from what is in use: the label set holds the class
        # list now, and guessing it from annotations was how an unpinned
        # project got one before there was anywhere to pin it
        classes = project.add_classes(names)
    except ProjectError as e:
        _error(str(e))
        raise typer.Exit(1) from None
    console.print(f"[green]Added {', '.join(names)}[/green] — classes: {', '.join(classes)}")

    _add_to_label_set(project, settings, classes)

    if push and project.label_studio.project_id is not None:
        client = _ls_client(settings, project, config_path)
        for name in names:
            try:
                client.add_class_to_config(project.label_studio.project_id, name)
            except LabelConfigError as e:
                _error(f"Label Studio config not updated: {e}")
                console.print(
                    "[yellow]project.toml is updated; add the class in the Label "
                    "Studio UI to match.[/yellow]"
                )
                raise typer.Exit(1) from None
        console.print(
            f"Label Studio project {project.label_studio.project_id} updated — "
            "refresh the tab to see it."
        )

    from strata.catalog import CatalogError

    catalog = _catalog_if_any(settings, project.catalog.name)
    if catalog is not None:
        try:
            label_set_id, _ = catalog.label_set(project.label_set_name)
        except CatalogError:
            return
        labelled = len(catalog.labelled(label_set_id, project.collections))
        skipped = len(catalog.skipped(label_set_id, project.collections))
        if labelled:
            console.print(
                f"[dim]{labelled} sample(s) were labelled before this class "
                f"existed.[/dim]"
            )
        if skipped:
            console.print(
                f"[dim]{skipped} skipped sample(s) may contain it — "
                f"'auto-labeller unskip' returns them to the queue.[/dim]"
            )


def _add_to_label_set(project: Project, settings, classes: list[str]) -> None:
    """Widen the catalog's label set to match the project's class list.

    Both have to move together: the labeling config Label Studio renders
    comes from project.toml, while what an export is validated against comes
    from the label set. A class in one and not the other means a reviewer can
    apply a label the catalog will then refuse.
    """
    from strata.catalog import CatalogError

    catalog = _catalog_if_any(settings, project.catalog.name)
    if catalog is None:
        return
    try:
        label_set_id, schema = catalog.label_set(project.label_set_name)
    except CatalogError:
        # No label set yet: ingest creates it from project.toml, so the
        # classes arrive with it
        return

    added = [c for c in classes if c not in schema.classes]
    if not added:
        return
    # Append-only, because a checkpoint maps output neurons to this list by
    # position and a run records the list it trained with
    catalog.set_classes(label_set_id, schema.model_copy(update={"classes": list(classes)}))
    console.print(f"Label set '{project.label_set_name}' widened by {', '.join(added)}")


@class_app.command("list")
def class_list(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
) -> None:
    """List the label set's classes with how many samples carry each."""
    project = _load_project(project_path)
    settings = _settings(config_path)
    catalog, _ = _catalog_for(settings, config_path, name=project.catalog.name)
    label_set_id, schema = _label_set_for(catalog, project)

    table = Table(title=f"Classes — {project.label_set_name}")
    table.add_column("Class", style="cyan")
    table.add_column("Samples", justify="right", style="green")
    for name in schema.classes:
        # From the class index rather than by decoding every annotation,
        # which is what that table exists for
        table.add_row(
            name, str(len(catalog.with_class(label_set_id, name, project.collections)))
        )
    console.print(table)

    declared = set(project.label_config.classes)
    drifted = set(schema.classes) ^ declared
    if declared and drifted:
        # Label Studio renders its config from project.toml while an export
        # is validated against the label set, so a class in one and not the
        # other lets a reviewer apply a label the catalog then refuses
        console.print(
            f"[yellow]project.toml and the label set disagree: "
            f"{', '.join(sorted(drifted))}[/yellow]"
        )


@app.command()
def unskip(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
    limit: int | None = typer.Option(None, help="Return only the first N skipped samples"),
) -> None:
    """Return skipped samples to the review queue.

    Skipping is how you park a sample whose content has no class yet, so
    after adding a class those samples are the best place to find examples
    of it. In Label Studio a skip is a cancelled annotation, deleted here so
    the task becomes reviewable again.
    """

    project = _load_project(project_path)
    settings = _settings(config_path)
    catalog, _ = _catalog_for(settings, config_path, name=project.catalog.name)
    label_set_id, _ = _label_set_for(catalog, project)

    skipped = catalog.skipped(label_set_id, project.collections)
    if not skipped:
        console.print("[yellow]Nothing is skipped.[/yellow]")
        return

    selected = skipped[:limit] if limit is not None else skipped

    ls_project_id = project.label_studio.project_id
    if ls_project_id is not None:
        task_map = _task_map(project, ls_project_id, catalog)
        task_ids = [task_map[s.id] for s in selected if s.id in task_map]
        if task_ids:
            client = _ls_client(settings, project, config_path)
            with console.status(f"Clearing the skip on {len(task_ids)} task(s)..."):
                client.delete_annotations(ls_project_id, task_ids)
            console.print(f"Cleared the skip on {len(task_ids)} Label Studio task(s)")
        if len(task_ids) < len(selected):
            # Ordinary rather than a fault: a skipped sample need never have
            # reached Label Studio, and push will create its task when it does
            console.print(
                f"[dim]{len(selected) - len(task_ids)} had no task there yet.[/dim]"
            )

    moved = catalog.unskip(label_set_id, [s.id for s in selected])
    console.print(
        f"[green]Returned {moved} sample(s) to the queue[/green] "
        f"({len(skipped) - moved} still skipped)"
    )


@app.command()
def init(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
    limit: int | None = typer.Option(
        None, help="Import only the first N samples (the rest arrive via push)"
    ),
) -> None:
    """Create a Label Studio project and fill it from the catalog.

    Everything already answered arrives answered, because Label Studio is a
    view of the catalog rather than a second copy of it. That is what makes
    a project disposable: delete it, run this again, lose nothing.
    """
    from .sync import save_task_map, tasks_to_push

    project = _load_project(project_path)
    settings = _settings(config_path)
    config = _catalog_config(settings, project.catalog.name)
    catalog, _ = _catalog_for(settings, config_path, name=project.catalog.name)
    label_set_id, label_schema = _label_set_for(catalog, project)

    if not label_schema.classes:
        _error("The label set declares no classes; add some before labelling.")
        raise typer.Exit(1)

    schema = _schema_for(project, catalog)
    # Answered first: with a limit, the point is to carry what is already
    # known rather than to fill the project with unreviewed samples
    samples = catalog.labelled(label_set_id, project.collections) + catalog.unlabelled(
        label_set_id, project.collections
    )
    if limit is not None:
        samples = samples[:limit]

    client = _ls_client(settings, project, config_path)
    ls_project_id = client.create_project(project.name)
    if not config.serve_url:
        # Only when Label Studio is the one reading files. Once tasks carry
        # signed URLs to the serving API, a local storage connection points
        # at a mount this deployment no longer has, and configuring one
        # would suggest the mount still matters.
        client.setup_local_storage(
            ls_project_id, path=f"/label-studio/data/{settings.label_studio.blobs_prefix}"
        )

    tasks, _ = tasks_to_push(
        samples, catalog, label_set_id, schema, _addressing(settings, config), {}
    )
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("Importing tasks", total=len(tasks))
        mapping = client.import_catalog_tasks(
            ls_project_id, tasks, on_progress=lambda n: progress.advance(bar, n)
        )
    save_task_map(project, ls_project_id, mapping, catalog.id)
    project.save_ls_project_id(settings.label_studio.url, ls_project_id)

    if len(mapping) != len(tasks):
        console.print(
            f"[yellow]{len(tasks) - len(mapping)} task(s) were created but could "
            f"not be mapped. Run 'auto-labeller push --rebuild-map' to recover "
            f"the mapping by listing them.[/yellow]"
        )

    answered = sum(1 for task in tasks if task.answered)
    console.print(
        f"[green]Created Label Studio project '{project.name}' "
        f"(ID: {ls_project_id}) with {len(tasks)} tasks[/green]"
    )
    console.print(f"  {answered} arrived already answered, {len(tasks) - answered} to review")
    console.print(f"  Classes: {', '.join(label_schema.classes)}")


@app.command()
def ingest(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
    batch: int = typer.Option(2000, help="Files per transaction"),
) -> None:
    """Scan the project's data root and register new files in the catalog.

    Which files count comes from the project's sample type, so a text
    project picks up documents where an image project picks up pictures.
    Grouping comes from the same place: frames of one video move together,
    so near-duplicates cannot straddle a train/val split. A corpus written
    by 'prepare' says which group each sample is in, and the type reads it.

    Registering is not queueing. The catalog holds the whole pool and `push`
    sends only what you are about to review, so there is no cost to
    cataloguing everything.
    """
    from strata.catalog import CatalogError

    project = _load_project(project_path)
    settings = _settings(config_path)
    data_dir = project.data_dir

    try:
        sample_type = project.sample_type()
    except ProjectError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    if not data_dir.exists():
        _error(f"Data root does not exist: {data_dir}")
        raise typer.Exit(1)

    catalog, catalog_root = _catalog_for(
        settings, config_path, create=True, name=project.catalog.name
    )
    try:
        label_set_id, _ = catalog.label_set(project.label_set_name)
    except CatalogError:
        label_set_id = catalog.create_label_set(
            project.label_set_name, project.schema.catalog_schema()
        )
        console.print(f"Created label set '{project.label_set_name}'")

    # Everything under the root, then checked. Filtering on the way in is how
    # a corpus ends up quietly smaller than the directory it came from.
    everything = [p for p in sorted(data_dir.rglob("*")) if p.is_file()]
    found = [p for p in everything if sample_type.allows(p)]
    skipped = [p for p in everything if p not in set(found)]

    if skipped:
        kinds = sorted({p.suffix.lower() or "(none)" for p in skipped})
        console.print(
            f"[yellow]{len(skipped):,} file(s) skipped — "
            f"{project.sample_type_name} does not admit {', '.join(kinds)}[/yellow]"
        )
    if not everything:
        # Nothing there yet is not a mistake: a project exists before its
        # data does, and this is what someone runs to find out.
        console.print(f"[yellow]No files under {data_dir} yet.[/yellow]")
        return
    if not found:
        # Files, and none of them admitted. Something is wrong — the wrong
        # folder, or a type that does not describe what is in it — and
        # returning quietly would report an empty corpus as a success.
        _error(
            f"None of the {len(everything):,} file(s) under {data_dir} are "
            f"{project.sample_type_name} "
            f"({', '.join('.' + e for e in sorted(sample_type.extensions))}). "
            f"Either the data is elsewhere, or [data] type names the wrong "
            f"thing for it."
        )
        raise typer.Exit(1)

    where = project.collections
    before = len(catalog.unlabelled(label_set_id, where)) + len(
        catalog.labelled(label_set_id, where)
    )
    # Grouped, because a group is one transaction and one group_id. Ungrouped
    # files share a bucket, so a plain image project is a handful of batches
    # rather than one per file.
    by_group: dict[str | None, list[Path]] = {}
    for path in found:
        by_group.setdefault(sample_type.group_id_for(path, data_dir), []).append(path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("Ingesting", total=len(found))
        for group_id, paths in by_group.items():
            for i in range(0, len(paths), batch):
                chunk = paths[i : i + batch]
                sources = {p: str(p.relative_to(data_dir)) for p in chunk}
                try:
                    catalog.ingest(
                        chunk,
                        media=sample_type.media,
                        subtype=type(sample_type).subtype(),
                        group_id=group_id,
                        # What only the type knows, plus where it came from
                        metadata_for=lambda p: {
                            "source_path": sources[p],
                            **sample_type.metadata_for(p, data_dir),
                        },
                        # Only for a type that has a canonical form; for the
                        # rest this is None and no file is read twice.
                        canonicalise=(
                            sample_type.canonicalise
                            if type(sample_type).canonicalises()
                            else None
                        ),
                        collections=project.collections,
                        on_sample=lambda _p: progress.advance(bar),
                    )
                except CatalogError as e:
                    # A file this type cannot store — a document in an
                    # encoding it will not guess at. The chunks before this
                    # one are committed, so re-running after fixing it
                    # carries on rather than starting over.
                    progress.stop()
                    _error(str(e))
                    raise typer.Exit(1) from None

    after = len(catalog.unlabelled(label_set_id, where)) + len(
        catalog.labelled(label_set_id, where)
    )
    skipped_count = len(catalog.skipped(label_set_id, where))
    console.print(
        f"[green]{len(found)} file(s) scanned, {after - before} new[/green] "
        f"into {catalog_root}"
    )
    console.print(
        f"  {after + skipped_count} sample(s) catalogued, "
        f"{len(by_group)} group(s) touched"
    )


@app.command()
def prepare(
    project_path: Path | None = ProjectOption,
    source: Path | None = typer.Option(
        None, "--from", help="Where the corpus is; defaults to the project's source_root"
    ),
    preparer_name: str = typer.Option(
        "", "--preparer", help="Which conversion to run; defaults to the project's setting"
    ),
) -> None:
    """Convert a corpus into the shape this project's sample type stores.

    Mail arrives as .eml, footage as video, and a catalog holds neither.
    This writes what it does hold — one document per message, one image per
    frame — into the project's data root, along with an index recording
    what the conversion knew: where each sample came from, and which video or thread
    it belongs to, so a group cannot straddle a train/val split.

    Then run 'ingest'. Two steps rather than one, because ingest is where
    content addressing, grouping and collections are decided, and a
    converter reaching around it would be a second implementation of the
    thing most worth having only one of.
    """
    from strata.catalog.preparers import PreparerError, available, for_source, resolve
    from strata.catalog.preparers import run as run_preparer

    project = _load_project(project_path)
    source_dir = source or project.source_dir
    if not source_dir.exists():
        _error(
            f"No corpus at {source_dir}. Put the files there, or point "
            f"[data] source_root at where they are."
        )
        raise typer.Exit(1)

    everything = [p for p in sorted(source_dir.rglob("*")) if p.is_file()]
    if not everything:
        console.print(f"[yellow]No files under {source_dir} yet.[/yellow]")
        return

    name = preparer_name or project.data.preparer
    try:
        wants = project.sample_type_name
        cls = resolve(name) if name else for_source(wants, everything[0])
    except (PreparerError, ProjectError) as e:
        _error(str(e))
        if not available():
            console.print(
                "[dim]Nothing is installed. A conversion is a plugin: "
                "uv pip install strata-prepare-email, or -video.[/dim]"
            )
        raise typer.Exit(1) from None

    if cls.produces != project.sample_type_name:
        # The corpus would convert, and ingest would then admit none of it
        _error(
            f"'{cls.name}' produces '{cls.produces}' samples and this project "
            f"ingests '{project.sample_type_name}'."
        )
        raise typer.Exit(1)

    preparer = cls()
    found = [p for p in everything if preparer.allows(p)]
    skipped = [p for p in everything if p not in set(found)]
    if skipped:
        kinds = sorted({p.suffix.lower() or "(none)" for p in skipped})
        console.print(
            f"[yellow]{len(skipped):,} file(s) skipped — '{cls.name}' does "
            f"not read {', '.join(kinds)}[/yellow]"
        )
    if not found:
        _error(
            f"None of the {len(everything):,} file(s) under {source_dir} are "
            f"read by '{cls.name}' "
            f"({', '.join('.' + e for e in sorted(cls.sources))})."
        )
        raise typer.Exit(1)

    out_dir = project.data_dir
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task(f"Preparing with '{cls.name}'", total=len(found))
        try:
            index = run_preparer(
                preparer, found, out_dir, on_source=lambda _p: progress.advance(bar)
            )
        except PreparerError as e:
            progress.stop()
            _error(str(e))
            raise typer.Exit(1) from None

    console.print(
        f"[green]{len(found):,} source file(s) → {len(index.samples):,} "
        f"sample(s)[/green] in {out_dir}"
    )
    # Anything left behind, said out loud. A corpus that arrives quietly
    # smaller than the source it came from is the failure ingest already
    # goes out of its way to avoid.
    for key, count in sorted(preparer.report().items()):
        console.print(f"  {key.replace('_', ' ')}: {count:,}")
    carrying = sum(1 for entry in index.samples.values() if entry.value is not None)
    if carrying:
        console.print(
            f"[dim]{carrying:,} sample(s) came with candidate annotations. They "
            f"are guesses, and nothing lands them in the catalog on its own."
            f"[/dim]"
        )
    console.print(f"\nNext: auto-labeller ingest --project {project.name}")


@app.command()
def train(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
    fresh: bool = typer.Option(
        False, "--fresh/--no-fresh", help="Cold start, ignoring the previous run"
    ),
    val_ratio: float = typer.Option(0.2, help="Share of samples to hold out"),
    job: str | None = typer.Option(
        None, "--job", help="Reattach to a round already running on the modelling host"
    ),
) -> None:
    """Train on the project's labelled data.

    A round freezes a dataset version in the catalog, materialises it, and
    trains from that directory — so the run resolves back to the exact
    samples behind it, and validation membership is inherited rather than
    recomputed.
    """
    project = _load_project(project_path)


    from .round import RoundError, describe, run_round

    settings = _settings(config_path)
    catalog, catalog_root = _catalog_for(settings, config_path, name=project.catalog.name)

    if job is not None:
        _reattach(settings, job)
        return
    if settings.modelling.url:
        _remote_round(project, catalog, settings, fresh=fresh, val_ratio=val_ratio)
        return

    try:
        # Materialising used to be a hard link away and over before anyone
        # looked. Pulling shards out of a bucket is minutes, and minutes of
        # nothing is indistinguishable from a hang.
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            bar = progress.add_task("Materialising", total=None)
            materialising = True

            def tick(done: int, total: int) -> None:
                nonlocal materialising
                progress.update(bar, completed=done, total=total)
                if materialising and done >= total:
                    # The last blob has landed and training is next, which
                    # is long, quiet and not this bar's business
                    materialising = False
                    progress.stop()
                    console.print(f"Materialised {total:,} sample(s). Training...")

            result = run_round(
                project,
                catalog,
                fresh=fresh,
                val_ratio=val_ratio,
                on_progress=tick,
                # Blobs already on this host, whatever the backend is. Every
                # version shares almost all its samples with the last, so
                # without this each one re-fetches a corpus sitting on disk.
                cache=catalog_root / "blobs",
            )
    except RoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    for line in describe(result):
        console.print(line)
    console.print(f"  checkpoint: {result.run.checkpoint}")


def _run_for_push(
    settings, project, store, run_id, remote: bool, catalog_id: str | None = None
) -> int | None:
    """Which run scores this push, in the numbering of whoever will score it.

    Run ids belong to the store that issued them. Asking a modelling host to
    predict with a local run id names a different model there, or none —
    silently, since both stores number from one.
    """
    if not remote:
        run = store.get(run_id) if run_id else store.latest(project.dataset_name, catalog_id)
        if run is None or not run.checkpoint:
            return None
        return run.id

    from .remote import RemoteError, Trainer

    trainer = Trainer(settings.modelling.url, settings.modelling.token)
    try:
        # Asked for by id or not, the host is the one that knows. Checking
        # now costs one request; not checking costs a pool fetched and
        # scored before anything notices.
        found = (
            trainer.run(run_id)
            if run_id
            else trainer.latest_run(project.dataset_name, catalog_id)
        )
    except RemoteError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    if found is None:
        return None
    if not found["run"].get("checkpoint"):
        _error(
            f"Run {found['run']['id']} on {settings.modelling.url} has no "
            f"checkpoint, so there is nothing to predict with."
        )
        raise typer.Exit(1)
    return found["run"]["id"]


def _remote_predictions(
    settings, run_id: str, checksums: list[str], features: dict | None = None
) -> dict:
    """Score a review pool on the host that has the GPU and the blobs.

    The same job machinery as a round, for the same reason: this is minutes
    of work over tens of thousands of samples, and a laptop that closes
    should not take it with it.
    """
    from pydantic import TypeAdapter

    from strata.labels import AnyPrediction
    from strata.modelling.service import PredictionRequest

    from .remote import RemoteError, Trainer

    if not settings.modelling.token:
        _error(
            "No token for the modelling host. Set $STRATA_MODELLING_TOKEN to "
            "the same value it was started with."
        )
        raise typer.Exit(1)

    trainer = Trainer(settings.modelling.url, settings.modelling.token)
    try:
        job = trainer.predict(
            PredictionRequest(run_id=run_id, checksums=checksums, features=features or {})
        )
    except RemoteError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    console.print(f"Scoring {len(checksums):,} sample(s) as job [bold]{job['id']}[/bold]")
    result = _follow(trainer, job["id"])

    if result.get("unknown"):
        console.print(
            f"[yellow]{len(result['unknown']):,} sample(s) the host's catalog "
            f"does not know — left out of the ranking[/yellow]"
        )
    # Through the union: a remote scoring pass returns whatever the task
    # emits, and reading spans as choices parses to an empty value rather
    # than failing — which would rank the queue by nothing at all.
    prediction = TypeAdapter(AnyPrediction)
    return {
        checksum: prediction.validate_python(value)
        for checksum, value in result.get("predictions", {}).items()
    }


def _reattach(settings, job_id: str) -> None:
    """Pick up a round that is already running elsewhere."""
    from .remote import Trainer

    if not settings.modelling.url:
        _error("No modelling host configured, so there is no job to reattach to.")
        raise typer.Exit(1)

    trainer = Trainer(settings.modelling.url, settings.modelling.token)
    result = _follow(trainer, job_id)
    run = result["run"]
    console.print(
        f"[green]Run {run['id']}[/green]"
        + (f", continuing run {run['parent_run_id']}" if run.get("parent_run_id") else " (cold)")
    )
    for metric, value in sorted(result.get("metrics", {}).items()):
        console.print(f"  {metric}: {value}")


def _follow(trainer, job_id: str) -> dict:
    """Watch a round to its end, surviving a network that comes and goes.

    Interrupting this stops watching, not training. That distinction is
    worth stating out loud, because Ctrl-C usually means the opposite.
    """
    from .remote import RemoteError

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        # Elapsed, because every stage looks identical while it is running
        # and the long one looks identical to a stalled one
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        bar = progress.add_task("Waiting for the host...")

        def show(job: dict) -> None:
            if job.get("state") == "unreachable":
                progress.update(
                    bar,
                    description=(
                        f"[yellow]Cannot reach the host (attempt "
                        f"{job['attempts']}) — the round is unaffected[/yellow]"
                    ),
                )
                return
            stage = job.get("stage", job.get("state", ""))
            if job.get("total"):
                stage = f"{stage} {job['done']:,}/{job['total']:,}"
            progress.update(bar, description=f"Host: {stage}")

        try:
            job = trainer.follow(job_id, on_state=show)
        except KeyboardInterrupt:
            progress.stop()
            console.print(
                f"[yellow]Stopped watching. The round is still running on the "
                f"host.[/yellow]\n  auto-labeller train --job {job_id}"
            )
            raise typer.Exit(0) from None
        except RemoteError as e:
            progress.stop()
            _error(str(e))
            raise typer.Exit(1) from None

    if job.get("state") == "failed":
        _error(f"The round failed on the host: {job.get('error')}")
        raise typer.Exit(1)
    return job["result"]


def _on_another_catalog(where: str, served: dict, catalog, config) -> str:
    """Why a machine on another catalog cannot work with this one, and what to change."""
    if served.get("id"):
        message = (
            f"{where} is on catalog {served.get('name') or '?'} ({served['id']}), "
            f"and this machine is on {catalog.id}."
        )
    else:
        why = served.get("error") or "none — it may predate machines saying which"
        message = f"{where} reports no catalog ({why}), and this machine is on {catalog.id}."
    if not config.url:
        message += (
            f" This catalog's index is SQLite under {config.root}, which only this "
            f"machine can read: the modelling host and the blob server open the index "
            f"themselves, so they can use this catalog only if they run here too. Give "
            f"it a Postgres url to share it, or unset [modelling] url to train here."
        )
    else:
        message += (
            " Make this catalog the default in that machine's config.toml, and "
            "restart its service."
        )
    return message


def _blob_server_catalog(url: str) -> dict:
    """What a blob server says it serves, from its /healthz."""
    import urllib.request

    with urllib.request.urlopen(f"{url.rstrip('/')}/healthz", timeout=10) as response:
        return json.loads(response.read()).get("catalog") or {}


def _remote_round(project, catalog, settings, fresh: bool, val_ratio: float) -> None:
    """Freeze a dataset here, and have another host train on it.

    The split is where the knowledge is. Which samples make a dataset is the
    project's business — its collections, its label set, its val ratio — and
    the catalog is reachable from both machines. Everything after that needs
    a GPU and the checkpoints, and both live there.
    """
    from strata.modelling.service import RoundRequest

    from .remote import RemoteError, Trainer

    if not settings.modelling.token:
        _error(
            "No token for the modelling host. Set $STRATA_MODELLING_TOKEN to "
            "the same value it was started with."
        )
        raise typer.Exit(1)

    label_set_id, _ = _label_set_for(catalog, project)
    labelled = catalog.labelled(label_set_id, project.collections)
    if not labelled:
        _error(
            f"Nothing is labelled for {project.label_set_name!r} in "
            f"{', '.join(project.collections)}, so there is nothing to train on."
        )
        raise typer.Exit(1)

    # Asked before anything is frozen. A host on another catalog would refuse
    # the round anyway; asking first says which machine to repoint, and
    # leaves no dataset version behind for a round that never ran.
    trainer = Trainer(settings.modelling.url, settings.modelling.token)
    try:
        served = trainer.served_catalog()
    except RemoteError as e:
        _error(str(e))
        raise typer.Exit(1) from None
    if served.get("id") != catalog.id:
        config = _catalog_config(settings, project.catalog.name)
        _error(_on_another_catalog("The modelling host", served, catalog, config))
        raise typer.Exit(1)

    dataset_id = catalog.create_dataset(
        project.dataset_name, label_set_id, collections=project.collections, val_ratio=val_ratio
    )
    ref = catalog.dataset_named(dataset_id)
    console.print(f"Dataset {ref.name} v{ref.version} → {settings.modelling.url}")

    try:
        job = trainer.submit(
            RoundRequest(
                dataset_id=dataset_id,
                # What the id means here, for the host to check it means the
                # same there: a copy of a catalog shares its numbering
                dataset_name=ref.name,
                dataset_version=ref.version,
                annotation_digest=ref.annotation_digest,
                catalog_id=catalog.id,
                model=project.model_ref,
                # Both sets, because only the host knows whether it has a
                # parent — and a cold run wants the longer schedule
                params=project.model.params,
                fresh_params=project.model.fresh_params,
                fresh=fresh,
                # Declarations only: the host reads the values out of the
                # catalog itself, as a local round does
                features=[spec.as_dict() for spec in project.feature_specs],
            )
        )
    except RemoteError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    # Printed before following, and printed plainly: from here the round is
    # the host's problem, and this id is how to ask after it from anywhere.
    console.print(f"Job [bold]{job['id']}[/bold] accepted. Training continues there.")
    console.print(f"  [dim]Reattach any time: auto-labeller train --job {job['id']}[/dim]\n")

    result = _follow(trainer, job["id"])
    run = result["run"]
    console.print(
        f"[green]Run {run['id']}[/green]"
        + (f", continuing run {run['parent_run_id']}" if run.get("parent_run_id") else " (cold)")
    )
    if result.get("materialised"):
        console.print(f"  [dim]{result['materialised']:,} sample(s) materialised there[/dim]")
    for metric, value in sorted(result.get("metrics", {}).items()):
        console.print(f"  {metric}: {value}")
    console.print(
        "[dim]The run and its checkpoint live on that host, which is where the "
        "next round will warm-start from.[/dim]"
    )


@app.command()
def push(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
    limit: int | None = typer.Option(
        None, help="Review only the top-N, most uncertain first"
    ),
    run_id: str | None = typer.Option(None, help="Predict with this run (default: latest)"),
    predictions: bool = typer.Option(
        True, "--predictions/--no-predictions", help="Attach pre-annotations"
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Replace existing predictions rather than adding to them"
    ),
    rebuild_map: bool = typer.Option(
        False, "--rebuild-map", help="Re-list tasks instead of trusting the local cache"
    ),
    strategy: str = typer.Option(
        "least-confident",
        "--strategy",
        help=(
            "How to order the queue: least-confident, margin or entropy for "
            "what teaches most per document; density for where a reviewer's "
            "hour is worth most, which is the early answer"
        ),
    ),
    empty_share: float = typer.Option(
        0.2,
        "--empty-share",
        help=(
            "How much of the batch may be samples the model found nothing in. "
            "They all score as maximally uncertain, so without a cap they take "
            "the whole queue."
        ),
    ),
) -> None:
    """Send unreviewed samples to Label Studio, least confident first.

    Predictions come from a recorded run, so what a reviewer sees is tied to
    a checkpoint that resolves back to the data behind it.
    """
    from strata.modelling import PredictRequest, RunStore
    from strata.modelling import predict as run_predict

    from .active_learning import STRATEGIES, certainty, rank
    from .adapter import prediction_to_results
    from .sync import rebuild_task_map, save_task_map, tasks_to_push

    if strategy not in STRATEGIES:
        _error(
            f"Unknown strategy {strategy!r}. Available: "
            f"{', '.join(sorted(STRATEGIES))}."
        )
        raise typer.Exit(1)

    project = _load_project(project_path)
    settings = _settings(config_path)
    catalog, catalog_root = _catalog_for(settings, config_path, name=project.catalog.name)
    label_set_id, _ = _label_set_for(catalog, project)
    addressing = _addressing(settings, _catalog_config(settings, project.catalog.name))
    schema = _schema_for(project, catalog)

    try:
        ls_project_id = project.require_ls_project_id(settings.label_studio.url)
    except ProjectError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    client = _ls_client(settings, project, config_path)
    task_map = _task_map(project, ls_project_id, catalog)
    if rebuild_map or not task_map:
        with console.status("Listing tasks in Label Studio..."):
            task_map, unrecognised = rebuild_task_map(
                client.list_tasks(ls_project_id), catalog, addressing, schema.data_key
            )
        if unrecognised:
            console.print(
                f"[yellow]{len(unrecognised)} task(s) point at nothing this catalog "
                f"knows — from before the cutover, or since removed.[/yellow]"
            )
        save_task_map(project, ls_project_id, task_map, catalog.id)

    pool = catalog.unlabelled(label_set_id, project.collections)
    if not pool:
        console.print("[yellow]Nothing is waiting for review.[/yellow]")
        return

    ranked, scored = pool, {}
    store = RunStore.local(project.runs_dir)
    remote = bool(settings.modelling.url)

    # Everything below deals in checksum -> ChoicesPrediction. The local
    # handler returns a ScoredPath wrapping one, and unwrapping it in some
    # places but not others is how a cache came to hold values that read
    # back empty.
    scores: dict[str, object] = {}
    scoring_run = _run_for_push(settings, project, store, run_id, remote, catalog.id)

    if predictions and scoring_run is not None:
        checksums = [s.checksum for s in pool]
        # What the model is to be told about each sample, and — the same
        # values — what keeps a cached answer the answer. A prediction is a
        # function of a checkpoint, some bytes and these; keyed on the
        # first two alone, a correction upstream would be served the score
        # it invalidated.
        specs = project.feature_specs
        resolved = catalog.features_for([s.id for s in pool], specs) if specs else {}
        by_checksum = {s.checksum: resolved.get(s.id, {}) for s in pool}
        if specs:
            uncovered = [s for s in pool if not by_checksum.get(s.checksum)]
            if uncovered:
                # Skipped and said out loud rather than scored on a blank.
                # The danger is the silence: a queue that only surfaces
                # covered samples never gets the rest labelled, so the gap
                # sustains itself.
                console.print(
                    f"[yellow]{len(uncovered):,} sample(s) carry no value for "
                    f"{', '.join(f.name for f in specs)} and cannot be scored — "
                    f"they stay out of this queue until they do.[/yellow]"
                )
                dropped = {s.checksum for s in uncovered}
                pool = [s for s in pool if s.checksum not in dropped]
                checksums = [s.checksum for s in pool]
        digests = {c: feature_digest(f) for c, f in by_checksum.items()}
        if remote:
            # The host keeps its own cache, keyed on its own run ids — which
            # is the only place that key means anything.
            scores = _remote_predictions(
                settings, scoring_run, checksums, by_checksum
            )
        else:
            from strata.modelling import PredictionCache

            cache = PredictionCache.local(project.runs_dir)
            scores = cache.get(scoring_run, checksums, digests)
            missing = [s for s in pool if s.checksum not in scores]
            if scores:
                console.print(
                    f"[dim]{len(scores):,} prediction(s) reused from run "
                    f"{scoring_run}; {len(missing):,} to make[/dim]"
                )
            if missing:
                with console.status(f"Predicting with run {scoring_run}..."):
                    fresh = run_predict(
                        PredictRequest(
                            run_id=scoring_run,
                            paths=_local_paths(catalog_root, missing),
                            features=[by_checksum[s.checksum] for s in missing],
                        ),
                        store,
                    )
                made = {s.checksum: p.value for s, p in zip(missing, fresh, strict=True)}
                cache.put(scoring_run, made, digests)
                scores.update(made)

        ranked = rank(pool, scores, STRATEGIES[strategy], empty_share=empty_share)
        scored = {s.id: scores[s.checksum] for s in ranked}
    elif predictions:
        console.print("[yellow]No run with a checkpoint yet; pushing without predictions.[/yellow]")

    # Added to the queue rather than reordered within it. A disputed sample
    # has an answer, so it is not unlabelled and would never appear — which
    # is the whole reason a conflict needs recording rather than leaving the
    # two answers to settle themselves.
    #
    # And first, ahead of the uncertainty ranking: where it would land there
    # depends on the model's opinion, which has no bearing on two people
    # disagreeing.
    conflicts = catalog.conflicts(label_set_id, project.collections)
    if conflicts:
        already = {s.id for s in ranked}
        disputed = [
            row
            for row in (catalog.by_checksum(c["checksum"]) for c in conflicts)
            if row is not None and row.id not in already
        ]
        if disputed:
            ranked = disputed + ranked
            console.print(
                f"[yellow]{len(disputed)} sample(s) were answered two ways — "
                f"pushed first so they are looked at again[/yellow]"
            )

    if limit is not None:
        ranked = ranked[:limit]

    tasks, report = tasks_to_push(
        ranked, catalog, label_set_id, schema, addressing, task_map
    )
    created = client.import_catalog_tasks(ls_project_id, tasks)
    task_map.update(created)
    save_task_map(project, ls_project_id, task_map, catalog.id)

    console.print(
        f"[green]{report.pushed} task(s) created[/green]"
        + (f", {report.already_present} already there" if report.already_present else "")
    )

    if scored:
        payload = [
            (s.id, prediction_to_results(scored[s.id], schema), certainty(scored[s.id]))
            for s in ranked
        ]
        pushed = client.push_catalog_predictions(
            ls_project_id,
            payload,
            task_map,
            model_version=f"run-{scoring_run}",
            replace_existing=refresh,
        )
        console.print(f"[green]{pushed} pre-annotation(s) attached from run {scoring_run}[/green]")


@app.command(name="export")
def export_annotations(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
    reviewed_only: bool = typer.Option(
        False,
        "--reviewed-only",
        help="Keep back answers nobody has opened, from a seeded project",
    ),
) -> None:
    """Pull corrected annotations out of Label Studio into the catalog.

    The catalog is what remembers; Label Studio is where the answering
    happens.

    Everything that comes back is recorded as a human answer, which is
    right when a person answered every task and wrong when the project was
    seeded from somewhere else: those tasks arrive already answered, and
    exporting halfway through a review stamps the seed's own guesses as
    ground truth. ``--reviewed-only`` keeps back anything nobody opened.
    """
    from .sync import pull_annotations

    project = _load_project(project_path)
    settings = _settings(config_path)
    catalog, _ = _catalog_for(settings, config_path, name=project.catalog.name)
    label_set_id, label_schema = _label_set_for(catalog, project)
    schema = _schema_for(project, catalog)

    try:
        ls_project_id = project.require_ls_project_id(settings.label_studio.url)
    except ProjectError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    client = _ls_client(settings, project, config_path)
    with console.status("Exporting from Label Studio (slow on large projects)..."):
        exported = client.export_raw(ls_project_id)

    items, report = pull_annotations(
        exported,
        catalog,
        label_set_id,
        schema,
        _addressing(settings, _catalog_config(settings, project.catalog.name)),
        label_schema.classes,
        reviewed_only=reviewed_only,
    )
    if report.untouched:
        console.print(
            f"[yellow]{report.untouched} task(s) were answered by an import "
            f"nobody has opened, and were left alone.[/yellow]"
        )

    if report.undeclared:
        # The catalog validates against the label set, so this would fail
        # partway through rather than at the end
        _error(
            f"Label(s) nobody declared: {', '.join(sorted(report.undeclared))}. "
            f"Add them with 'auto-labeller class add', then export again."
        )
        raise typer.Exit(1)

    written = catalog.annotate_many(label_set_id, items, source="human")
    console.print(
        f"[green]{written.annotated} annotation(s) and {written.skipped} skip(s) "
        f"into the catalog[/green]"
    )
    if report.unrecognised:
        console.print(
            f"[yellow]{len(report.unrecognised)} task(s) point at nothing this "
            f"catalog knows, and were left alone.[/yellow]"
        )


#: What "how is it going" means per task. A model may report anything it
#: likes alongside; this is only which one the history plots by default.
HEADLINE_METRIC = {
    "classification": "val_accuracy",
    "span": "val_span_f1",
}


@app.command()
def report(
    project_path: Path | None = ProjectOption,
    metric: str | None = typer.Option(
        None, help="Which metric to plot (default: this task's headline one)"
    ),
    run_id: str | None = typer.Option(
        None, "--run", help="Detail one run instead of the history"
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit the same thing as JSON, for a chart or a script"
    ),
) -> None:
    """Show the training history, or one run in detail.

    Read from the run store rather than from round folders, so a metric
    across rounds is one query.

    ``--json`` writes the same query to stdout as JSON and nothing else, so
    it pipes. It carries more than the table does — every metric rather than
    the one plotted, and each run's ``params`` and ``classes`` — because the
    table's job is to be read and this one's is to be computed on. Those two
    fields are what let a consumer decide whether two runs are even
    comparable: a project whose ``[model.params]`` reshape the task can
    produce a history that looks like progress and is not, and the table's
    delta column cannot tell.

    A change is shown only where one run actually continues the one above:
    a warm-started number means something against its parent and nothing
    against a run from another lineage. Rows marked unchained continue
    nothing in the store — either a genuine cold start, or a round imported
    from before the store existed, whose lineage was never recorded.
    """
    from strata.modelling import RunStore

    project = _load_project(project_path)
    if not (project.runs_dir / "runs.db").exists():
        _error(f"No runs recorded at {project.runs_dir}. Run 'train' first.")
        raise typer.Exit(1)

    store = RunStore.local(project.runs_dir)

    # The headline number differs by task, and defaulting to the
    # classification one meant this command reported nothing at all for a
    # span project — every run had metrics, just not that name.
    metric = metric or HEADLINE_METRIC.get(project.schema.task, "val_accuracy")

    if run_id is not None:
        run = store.get(run_id)
        if run is None:
            _error(f"No run with id {run_id}")
            raise typer.Exit(1)
        if as_json:
            _emit_json(
                {
                    "run": _run_json(run),
                    "chain": [_run_json(r) for r in store.chain(run.id)],
                    # Only here, never in the history: there are as many of
                    # these as epochs times metrics, and the history is a
                    # list of runs rather than a list of curves.
                    "curve": [
                        {"epoch": epoch, **reported}
                        for epoch, reported in store.curve(run.id)
                    ],
                }
            )
            return
        _print_run(store, run)
        return

    history = store.history(project.dataset_name, metric)
    if not history:
        _error(
            f"No run recorded {metric!r} for '{project.dataset_name}'. "
            + (
                f"Try --metric {', --metric '.join(available)}, "
                if (available := store.metric_names(project.dataset_name))
                else ""
            )
            + "or --run to inspect one."
        )
        raise typer.Exit(1)

    # Computed once, rendered twice. The delta rule below is the whole
    # reason this is not a plain dump of the store, and a JSON consumer
    # reimplementing it from the raw rows would get it subtly wrong.
    rows: list[dict] = []
    seen: dict[int, float] = {}
    versions: dict[int, int] = {}
    for run_num, version, value in history:
        run = store.get(run_num)
        parent = run.parent_run_id
        # Against its own parent, and only when both were scored on the same
        # held-out samples. A dataset version going backwards means the
        # lineage crossed into the catalog from the old layout, where the
        # split was recomputed every round — comparing those produced a
        # +0.10 that measured nothing but a change of validation set.
        before, now = versions.get(parent), version
        comparable = (
            parent in seen
            and before is not None
            and now is not None
            and before <= now
        )
        rows.append(
            {
                "run": run,
                "value": value,
                "version": version,
                "delta": (value - seen[parent]) if comparable else None,
                "warm": bool(parent),
            }
        )
        seen[run_num] = value
        versions[run_num] = version

    if as_json:
        _emit_json(
            {
                "dataset": project.dataset_name,
                "metric": metric,
                "runs": [
                    {
                        **_run_json(row["run"]),
                        "value": row["value"],
                        # Null rather than absent, and null rather than zero:
                        # "this cannot be compared to its parent" is not the
                        # same statement as "it did not move".
                        "delta": row["delta"],
                        "lineage": "warm" if row["warm"] else "unchained",
                    }
                    for row in rows
                ],
            }
        )
        return

    table = Table(title=f"{project.dataset_name} — {metric}")
    table.add_column("Run", justify="right")
    table.add_column("Dataset", justify="right")
    table.add_column(metric, justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("Lineage")
    for row in rows:
        version = row["version"]
        # Whether it continued, not what from. An id is a timestamp and a
        # host now, and two of them in one row of a table is a row of
        # ellipses — the chain itself is what `report --run` is for.
        lineage = "warm" if row["warm"] else "[yellow]unchained[/yellow]"
        shown = f"v{version}" if version is not None else "[dim]—[/dim]"
        delta = f"{row['delta']:+.4f}" if row["delta"] is not None else ""
        table.add_row(
            _short(row["run"].id), shown, f"{row['value']:.4f}", delta, lineage
        )
    console.print(table)


def _short(run_id) -> str:
    """A run id without its microseconds, for showing a person.

    Full ids are what everything keys on; the microseconds are the part
    nobody reads, and a table of thirty-character strings is a table nobody
    reads either.
    """
    stamp, _, host = str(run_id).partition("-")
    return f"{stamp[:15]}-{host}" if host else str(run_id)


def _run_json(run) -> dict:
    """One run, as something to compute on rather than to read.

    Everything the store holds, including ``params`` and ``classes``. Those
    are the fields that answer whether two runs are asking the same
    question: a metric moved by changing the data, the model, or what the
    model was told to do, and only the last of those is invisible in a
    table of numbers.
    """
    data = run.model_dump(mode="json")
    # Not a field on the model, and the one thing a caller would otherwise
    # have to reimplement the id format to get.
    data["short"] = run.short
    return data


def _emit_json(payload: dict) -> None:
    """Straight to stdout, past rich.

    ``console.print`` would wrap it to the terminal width and colour it,
    which is right for a table and fatal for something being piped into
    ``jq``. Written with ``print`` for the same reason the width is not
    consulted: this output has no reader to be considerate of.
    """
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _print_run(store, run) -> None:
    console.print(f"[bold]Run {run.short}[/bold] — {run.model} v{run.model_version}")
    version = f"v{run.dataset_version}" if run.dataset_version is not None else (
        "no dataset version — imported from before the catalog"
    )
    console.print(f"  dataset:   {run.dataset} {version}")
    console.print(f"  label set: {run.label_set}")
    console.print(f"  classes:   {', '.join(run.classes)}")
    console.print(f"  params:    {run.params}")
    console.print(f"  checkpoint: {run.checkpoint}")

    chain = store.chain(run.id)
    if len(chain) > 1:
        console.print(f"  continues: {' -> '.join(r.short for r in chain)}")
    else:
        console.print("  unchained — continues nothing in the store")

    curve = store.curve(run.id)
    if curve:
        console.print(
            f"  curve:     {len(curve)} epoch(s) recorded — "
            f"'report --run {run.short} --json' has them"
        )

    if run.metrics:
        table = Table("Metric", "Value", box=None, pad_edge=False)
        for name, value in sorted(run.metrics.items()):
            table.add_row(name, f"{value:.4f}")
        console.print(table)


@app.command(name="catalog-stats")
def catalog_stats(
    catalog_name: str = CatalogOption,
    config_path: Path = ConfigOption,
) -> None:
    """What is in a catalog: samples, label sets, and the class breakdown.

    The per-class counts come from the annotation_class index rather than
    from scanning stored values, so they are also the check that indexing
    did its job.
    """
    from sqlalchemy import func, select

    from strata.catalog import EVERYTHING
    from strata.catalog import tables as t

    settings = _settings(config_path)
    catalog, catalog_root = _catalog_for(settings, config_path, name=catalog_name)
    config = _catalog_config(settings, catalog_name)
    with catalog.engine.connect() as conn:
        total = conn.execute(select(func.count()).select_from(t.sample)).scalar()
        groups = conn.execute(
            select(func.count(func.distinct(t.sample.c.group_id))).where(
                t.sample.c.group_id.is_not(None)
            )
        ).scalar()
        ungrouped = conn.execute(
            select(func.count()).select_from(t.sample).where(t.sample.c.group_id.is_(None))
        ).scalar()
        sizes = [
            row.n
            for row in conn.execute(
                select(func.count().label("n"))
                .select_from(t.sample)
                .where(t.sample.c.group_id.is_not(None))
                .group_by(t.sample.c.group_id)
            )
        ]
        collections = conn.execute(
            select(t.sample_collection.c.collection, func.count())
            .group_by(t.sample_collection.c.collection)
            .order_by(t.sample_collection.c.collection)
        ).all()
        uncollected = conn.execute(
            select(func.count())
            .select_from(t.sample)
            .outerjoin(
                t.sample_collection,
                t.sample_collection.c.sample_id == t.sample.c.id,
            )
            .where(t.sample_collection.c.sample_id.is_(None))
        ).scalar()
        label_sets = conn.execute(select(t.label_set.c.id, t.label_set.c.name)).all()

    where = _redacted(config.url) if config.url else catalog_root
    console.print(f"[bold]{where}[/bold]: {total} sample(s)")
    if groups:
        console.print(f"  {groups} group(s), {ungrouped} sample(s) in no group")
        sizes.sort()
        singletons = sum(1 for n in sizes if n == 1)
        # A group is indivisible, so the largest one is the floor on how
        # coarse the split can be; singletons are groups doing no work
        console.print(
            f"    {sizes[0]}–{sizes[-1]} samples per group "
            f"(median {sizes[len(sizes) // 2]}), "
            f"largest is {sizes[-1] / max(total, 1):.1%} of the catalog"
        )
        if singletons:
            console.print(
                f"    {singletons} group(s) hold a single sample, so grouping "
                f"changes nothing for them"
            )
    elif total:
        # Always said, because no grouping is the answer worth noticing: for
        # video frames it means near-duplicates will be split individually
        # and validation will score the model on what it trained on.
        console.print(
            "  [yellow]no grouping[/yellow] — every sample is its own group, "
            "which is right for standalone images and wrong for video frames"
        )

    if collections:
        console.print("\n[bold]collections[/bold]")
        for name, count in collections:
            console.print(f"  {name:<40} {count:>9,}")
    if uncollected:
        # Reachable only through EVERYTHING, so no project would ever see
        # them — worth saying rather than leaving them to be discovered
        console.print(
            f"  [yellow]{uncollected:,} sample(s) in no collection[/yellow] — "
            f"no project draws from them"
        )

    if not label_sets:
        console.print("\n[yellow]No label sets yet.[/yellow]")
        return

    for label_set_id, name in label_sets:
        _, schema = catalog.label_set(name)
        # The whole catalog on purpose: this is the view of everything there
        # is, not of what any one job draws from
        labelled = catalog.labelled(label_set_id, EVERYTHING)
        queue = catalog.unlabelled(label_set_id, EVERYTHING)
        # Whether choices are exclusive is a classification detail. A span
        # or bbox label set has no such notion, and reading it off one
        # raised here rather than printing — so this command worked for
        # every image project and for no text one.
        multiple = getattr(schema, "multiple", None)
        detail = "" if multiple is None else f", {'multi' if multiple else 'single'}-choice"
        console.print(f"\n[bold]{name}[/bold] — {schema.task}{detail}")
        console.print(f"  {len(labelled)} annotated, {len(queue)} awaiting review")

        table = Table("Class", "Samples", box=None, pad_edge=False)
        for class_name in schema.classes:
            table.add_row(
                class_name,
                str(len(catalog.with_class(label_set_id, class_name, EVERYTHING))),
            )
        console.print(table)


@app.command(name="import-rounds")
def import_rounds_command(
    project_path: Path | None = ProjectOption,
    chain: bool = typer.Option(
        False,
        "--chain",
        help="Record each round as continuing the last. Only if they really were "
        "warm-started — nothing on disk says so.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report without writing"),
) -> None:
    """Carry rounds/*/metadata.json into the run store.

    The pre-catalog rounds hold the only record of how the model got where it
    is. Checkpoints are referenced where they sit rather than copied.
    """
    from .import_rounds import describe, import_rounds, read_rounds

    project = _load_project(project_path)
    rounds = read_rounds(project)
    if not rounds:
        console.print(f"[yellow]No rounds found under {project.rounds_dir}[/yellow]")
        raise typer.Exit()

    if dry_run:
        console.print(f"Would import {len(rounds)} round(s):")
        for metadata in rounds:
            metrics = ", ".join(
                f"{k}={v}" for k, v in (metadata.get("metrics") or {}).items()
            )
            console.print(
                f"  round {metadata.get('round')}: "
                f"{len(metadata.get('classes') or [])} classes"
                + (f", {metrics}" if metrics else "")
            )
        return

    from strata.modelling import RunStore

    store = RunStore.local(project.runs_dir)
    if store.latest(project.dataset_name) is not None:
        # History reads in run order, so importing after a catalog round has
        # already been recorded files the older rounds after the newer ones
        console.print(
            "[yellow]This project already has runs. Importing now files the "
            "historical rounds after them, since the curve reads in run "
            "order.[/yellow]"
        )

    report = import_rounds(project, store, chain=chain)
    for line in describe(report, chained=chain):
        console.print(line)


@app.command(name="runs-merge")
def runs_merge(
    from_dir: Path = typer.Option(
        ..., "--from", help="A runs directory: runs.db and its checkpoints"
    ),
    checkpoints: bool = typer.Option(
        False, "--checkpoints", help="Copy the checkpoint files too. They are large."
    ),
    apply: bool = typer.Option(
        False, "--apply", help="Write. Without this, the merge is only reported."
    ),
    project_path: Path | None = ProjectOption,
) -> None:
    """Fold another run store into this project's.

    A project keeps runs beside its own checkpoints and a modelling host
    keeps its own, so a project trained on both has its history split in
    two. This puts it back together, which is what `report` needs to draw
    one curve.

    Run ids carry the time and the host that made them, so nothing
    collides and a run already here is skipped rather than duplicated —
    run this twice and the second one does nothing.

    Checkpoints stay where they are unless --checkpoints is given: they are
    the large half, and a merge is usually about reading a history rather
    than training from it. A run whose checkpoint did not come is recorded
    as having none, so nothing later tries to warm-start from a file that
    is not there.
    """
    from strata.modelling import RunStore, StoreMergeError, merge_stores

    project = _load_project(project_path)
    source_dir = Path(from_dir)
    if not (source_dir / "runs.db").exists():
        _error(f"No runs.db under {source_dir}.")
        raise typer.Exit(1)

    source = RunStore.local(source_dir)
    target = RunStore.local(project.runs_dir)

    try:
        report = merge_stores(
            source, target, checkpoints=checkpoints, dry_run=not apply
        )
    except StoreMergeError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    if not report.runs and not report.already_present:
        console.print(f"No runs in {source_dir}.")
        return

    for line in report.lines():
        console.print(f"  {line}")
    if report.orphaned:
        console.print(
            "[yellow]Some runs continued from a run in neither store. They "
            "were copied without the link, so they read as cold.[/yellow]"
        )
    if not apply:
        console.print("[yellow]Nothing was written. Re-run with --apply.[/yellow]")
        return
    console.print(f"[green]{report.runs} run(s) merged[/green]")
    if not checkpoints and report.runs:
        console.print(
            "[dim]Checkpoints were left behind, so the merged runs cannot be "
            "trained or predicted from here. Pass --checkpoints if you need "
            "them.[/dim]"
        )


@app.command(name="catalog-merge")
def catalog_merge(
    from_url: str = typer.Option(
        ..., "--from", help="Index URL of the copy whose answers to fold in"
    ),
    apply: bool = typer.Option(
        False, "--apply", help="Write. Without this, the merge is only reported."
    ),
    catalog_name: str = CatalogOption,
    config_path: Path = ConfigOption,
) -> None:
    """Fold a copy's annotations back into this host's catalog.

    For work done away from the index: take a copy with `catalog-copy`,
    label offline, bring the answers home. Only annotations move — the
    samples are already here, since a copy is the same corpus.

    Where both sides answered the same sample differently, this catalog
    keeps what it had and the disagreement is recorded for a person to look
    at. Nothing is overwritten and nothing is discarded.

    Reports without writing unless --apply is given. Conflicts are the
    interesting outcome and are much easier to read before the merge than to
    find after it.
    """
    from strata.catalog import Catalog, MergeError, merge_annotations
    from strata.catalog.config import blobs_for

    settings = _settings(config_path)
    target, _ = _catalog_for(settings, config_path, name=catalog_name)
    # A copy is the same corpus, so it reads the same bytes this catalog does
    source = Catalog.connect(from_url, blobs_for(_catalog_config(settings, catalog_name)))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed:,}/{task.total:,}"),
            console=console,
        ) as progress:
            bar = progress.add_task("Merging", total=1)

            def tick(done: int, total: int) -> None:
                progress.update(bar, completed=done, total=total)

            report = merge_annotations(source, target, dry_run=not apply, on_progress=tick)
    except MergeError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    if not report.total and not report.unknown_label_sets:
        console.print(f"Nothing to merge from {from_url}.")
        return

    for line in report.lines():
        console.print(f"  {line}")
    if report.unknown_samples:
        console.print(
            "[yellow]Some answers are for samples this catalog has never "
            "seen. They were ingested on the other machine after the copy "
            "was taken; ingest those files here and merge again.[/yellow]"
        )
    if not apply:
        console.print("[yellow]Nothing was written. Re-run with --apply.[/yellow]")
        return
    console.print(f"[green]{report.written:,} annotation(s) merged[/green]")
    if report.conflicted:
        console.print(
            f"[yellow]{report.conflicted:,} sample(s) were answered both ways. "
            f"This catalog kept its own; the next 'push' sends them for "
            f"review ahead of everything else.[/yellow]"
        )


@app.command(name="catalog-copy")
def catalog_copy(
    to_url: str = typer.Option(..., "--to", help="Index URL to copy into"),
    catalog_name: str = CatalogOption,
    config_path: Path = ConfigOption,
) -> None:
    """Copy this host's catalog index into another database.

    Only the index moves. Blobs are addressed by content, so the catalog
    keeps pointing at exactly the same bytes — changing database is six
    tables, not a data migration.

    Sample ids are preserved. Every annotation, every dataset member and the
    Label Studio task map are keyed on them, so renumbering would silently
    repoint every task at a different image.
    """
    from strata.catalog import Catalog, CopyError, copy_index
    from strata.catalog.config import blobs_for

    settings = _settings(config_path)
    source, _ = _catalog_for(settings, config_path, name=catalog_name)
    # Only the index moves: the copy points at exactly the same bytes
    target = Catalog.connect(to_url, blobs_for(_catalog_config(settings, catalog_name)))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            bar = progress.add_task("Copying")

            def tick(table: str, count: int) -> None:
                progress.update(bar, description=f"Copying {table} ({count:,})")

            report = copy_index(source, target, on_progress=tick)
    except CopyError as e:
        # Outside the live display, or the message lands under a spinner
        # that never gets a chance to clear
        _error(str(e))
        raise typer.Exit(1) from None

    for name, count in report.copied.items():
        console.print(f"  {name:<18} {count:>9,}")
    console.print(f"[green]{report.total:,} row(s) copied[/green]")
    console.print(
        "[dim]Blobs were not touched. Point [catalog] url at the new index "
        "and leave root as it is.[/dim]"
    )


@app.command()
def relink(
    project_path: Path = ProjectOption,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would change; touch nothing"
    ),
    config_path: Path = ConfigOption,
) -> None:
    """Repoint a project's Label Studio tasks at their current image URLs.

    Two uses, one operation. It moves existing tasks off the local blob
    mount and onto the serving API, which is what lets the mount go away.
    And it re-signs: a signed URL expires, so a task left in a review queue
    longer than a signature's life stops loading, and this is the fix.

    Tasks whose URL names no sample are left alone and reported. A task made
    before the catalog points at a real image that nothing here can identify,
    and rewriting it would destroy the only record of what it showed.
    """
    from .sync import relink as plan_relink

    project = _load_project(project_path)
    settings = _settings(config_path)
    catalog, _ = _catalog_for(settings, config_path, name=project.catalog.name)
    addressing = _addressing(settings, _catalog_config(settings, project.catalog.name))
    schema = _schema_for(project, catalog)

    client = _ls_client(settings, project, config_path)
    try:
        ls_project_id = project.require_ls_project_id(settings.label_studio.url)
    except ProjectError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    where = addressing.base_url or f"the {addressing.prefix} mount"
    console.print(f"[bold]Label Studio project {ls_project_id}[/bold] → {where}\n")

    with console.status("Listing tasks..."):
        tasks = client.list_tasks(ls_project_id)
    report = plan_relink(tasks, catalog, addressing, schema.data_key)

    console.print(f"  {report.total:,} task(s): {len(report.changes):,} to repoint, "
                  f"{report.unchanged:,} already current")
    if report.unrecognised:
        console.print(
            f"  [yellow]{len(report.unrecognised):,} name no sample and are "
            f"left alone[/yellow]"
        )
        for url in report.unrecognised[:3]:
            console.print(f"    [dim]{escape(url)}[/dim]")

    if dry_run:
        console.print("\n[dim]Nothing was changed.[/dim]")
        return
    if not report.changes:
        console.print("\n[green]Nothing to do.[/green]")
        return

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("Repointing", total=len(report.changes))
        done = 0
        for task_id, data in report.changes:
            client.update_task_data(task_id, data)
            done += 1
            progress.update(bar, advance=1)

    console.print(f"[green]{done:,} task(s) repointed[/green]")
    if addressing.base_url:
        console.print(
            "[dim]These URLs carry an expiry. Run this again if a queue sits "
            "long enough for images to stop loading.[/dim]"
        )


@app.command(name="catalog-repack")
def catalog_repack(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would move; write nothing"
    ),
    shard_mb: int = typer.Option(
        512, "--shard-mb", help="Shard size in MB (a shard is held in memory while packed)"
    ),
    verify: int = typer.Option(64, help="Members to read back and check afterwards"),
    catalog_name: str = CatalogOption,
    config_path: Path = ConfigOption,
) -> None:
    """Pack local blobs into tar shards in object storage.

    The cutover to shared storage. Bytes are copied into the bucket and the
    index is repointed; nothing local is deleted, because those files are
    still what Label Studio serves images from and they are the way back if
    this goes wrong.

    Safe to interrupt and re-run: a sample already in a shard is skipped, so
    a second run resumes rather than packing twice.
    """
    from strata.catalog import LocalBackend, RepackError, repack_blobs
    from strata.catalog.config import blobs_for

    settings = _settings(config_path)
    config = _catalog_config(settings, catalog_name)
    if not config.s3_endpoint:
        _error(
            "No object storage configured for this catalog. Set s3_endpoint and "
            "s3_bucket in its [catalog] table, with the credentials in "
            "$STRATA_S3_ACCESS_KEY and $STRATA_S3_SECRET_KEY — this packs blobs "
            "into a bucket, so there is nowhere to put them otherwise."
        )
        raise typer.Exit(1)

    catalog, root = _catalog_for(settings, config_path, name=catalog_name)
    # This catalog's bucket, not the host default's: with --catalog naming
    # another, the default's would pack one corpus into someone else's.
    target = blobs_for(config)
    # Before any put, since the size is read when a shard is opened. Larger
    # shards mean fewer objects and fewer requests; the cost is memory, as a
    # shard is buffered whole and then read into one bytes object to upload.
    target.shard_bytes = shard_mb * 1024 * 1024
    # Explicitly the local one: blobs_for answers with the bucket once an
    # endpoint is set, and that is the destination, not the source.
    source = LocalBackend(root / "blobs")

    console.print(f"[bold]from[/bold]  {source.root}")
    console.print(
        f"[bold]to[/bold]    {config.s3_endpoint} "
        f"bucket={config.s3_bucket} shards={shard_mb} MB\n"
    )

    try:
        if dry_run:
            report = repack_blobs(catalog, target, source=source, dry_run=True)
            shards = -(-report.bytes // target.shard_bytes) if report.bytes else 0
            console.print(
                f"  {report.samples:,} sample(s) to pack, "
                f"{report.bytes / 1e9:.1f} GB, about {shards} shard(s)"
            )
            if report.already_packed:
                console.print(f"  [dim]{report.already_packed:,} already packed[/dim]")
            console.print("\n[dim]Nothing was written.[/dim]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            bar = progress.add_task("Packing")

            def tick(report) -> None:
                progress.update(
                    bar,
                    description=(
                        f"Packing — {report.samples:,} sample(s), "
                        f"{report.bytes / 1e9:.1f} GB, "
                        f"{len(report.shards)} shard(s) uploaded"
                    ),
                )

            report = repack_blobs(
                catalog, target, source=source, verify=verify, on_progress=tick
            )
    except RepackError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    console.print(
        f"[green]{report.samples:,} sample(s) packed[/green] into "
        f"{len(report.shards)} shard(s), {report.bytes / 1e9:.1f} GB"
    )
    if report.already_packed:
        console.print(f"  [dim]{report.already_packed:,} were already packed[/dim]")

    if report.failures:
        console.print(f"\n[red]{len(report.failures)} of {report.verified} "
                      f"verified member(s) read back wrong[/red]")
        for line in report.failures[:5]:
            console.print(f"  {line}")
        console.print(
            "\n[dim]The index now points at these shards. Local blobs are "
            "untouched, so reverting means restoring location/offset/length "
            "from a backup of the index.[/dim]"
        )
        raise typer.Exit(1)

    console.print(f"  [dim]{report.verified} member(s) verified[/dim]")
    # escape, or Rich reads the TOML section name as a markup tag and drops
    # it — leaving advice that does not say which section
    console.print(
        "\n[bold]" + escape("Set [catalog] s3_endpoint and s3_bucket in "
                            "config.toml now.") + "[/bold]"
    )
    console.print(
        "[dim]Not optional: the index points at shards, and the local backend "
        "would look for one as a file and not find it. Keep the local blobs "
        "even so — Label Studio serves images straight off that mount rather "
        "than through the catalog, so it is unaffected either way.[/dim]"
    )


@app.command(name="catalog-check")
def catalog_check(
    catalog_name: str = CatalogOption,
    config_path: Path = ConfigOption,
) -> None:
    """Ask the blob server and the modelling host which catalog they are on.

    Each machine reads its own config.toml, so pointing the setup at another
    catalog is an edit on each, and a machine missed is what this catches: a
    blob server left on the old catalog answers 404 for every new sample, and
    a modelling host left on it refuses every round.
    """
    from .remote import RemoteError, Trainer

    settings = _settings(config_path)
    config = _catalog_config(settings, catalog_name)
    catalog = _catalog_if_any(settings, catalog_name)
    if catalog is None:
        _error(f"No catalog at {config.root} yet, so there is nothing to compare against.")
        raise typer.Exit(1)
    name = catalog_name or settings.catalogs.default_name or "default"
    console.print(f"[bold]this machine[/bold]  {escape(name)}  {catalog.id}")

    hosts = []
    if config.serve_url:
        hosts.append(
            ("blob server", config.serve_url, lambda: _blob_server_catalog(config.serve_url))
        )
    if settings.modelling.url:
        trainer = Trainer(settings.modelling.url, settings.modelling.token)
        hosts.append(("modelling host", settings.modelling.url, trainer.served_catalog))
    if not hosts:
        console.print(
            escape("No serve_url and no [modelling] url, so no other machine reads a catalog.")
        )
        return

    ok = True
    for label, url, ask in hosts:
        try:
            served = ask()
        except (RemoteError, OSError, ValueError) as e:
            ok = False
            console.print(f"[bold]{label}[/bold]  {url}  [red]unreachable[/red] — {escape(str(e))}")
            continue
        if served.get("id") == catalog.id:
            console.print(f"[bold]{label}[/bold]  {url}  [green]same catalog[/green]")
        else:
            ok = False
            console.print(f"[bold]{label}[/bold]  {url}  [red]another catalog[/red]")
            _error(_on_another_catalog(f"The {label} at {url}", served, catalog, config))
    raise typer.Exit(0 if ok else 1)


@app.command(name="catalog-probe")
def catalog_probe(
    catalog_name: str = CatalogOption,
    config_path: Path = ConfigOption,
) -> None:
    """Check that this host can actually reach its catalog.

    Reports what is configured, then proves it: the index answers a query,
    and a blob written comes back byte for byte. The round trip is the part
    worth having — object storage that ignores a Range header returns the
    start of the shard for every sample, which reads as data rather than as
    an error.

    Writes into a probe prefix and removes it afterwards, so nothing lands
    among real shards.
    """
    import uuid

    from sqlalchemy import func, select

    from strata.catalog import checksum_of
    from strata.catalog import tables as t
    from strata.catalog.config import blobs_for

    settings = _settings(config_path)
    config = _catalog_config(settings, catalog_name)
    ok = True

    # -- the index -----------------------------------------------------
    where = (
        _redacted(config.url)
        if config.url
        else f"sqlite under {config.root}"
    )
    console.print(f"[bold]index[/bold]  {where}")
    try:
        catalog, root = _catalog_for(settings, config_path, create=True, name=catalog_name)
        config = _catalog_config(settings, catalog_name)
        with catalog.engine.connect() as conn:
            samples = conn.execute(select(func.count()).select_from(t.sample)).scalar()
        console.print(f"  [green]reachable[/green] — {samples:,} sample(s)")
    except Exception as e:
        console.print(f"  [red]unreachable[/red] — {type(e).__name__}: {e}")
        raise typer.Exit(1) from None

    # -- the blobs -----------------------------------------------------
    endpoint = config.s3_endpoint
    console.print(
        "\n[bold]blobs[/bold]  "
        + (f"{endpoint} bucket={config.s3_bucket}" if endpoint
           else f"files under {root / 'blobs'}")
    )

    body = bytes(range(256)) * 64
    probe = root / f".probe-{uuid.uuid4().hex[:8]}"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_bytes(body)
    uploaded = None
    try:
        blobs = blobs_for(config)
        if endpoint:
            # A prefix of its own, so a probe never lands among real shards
            blobs.prefix = "probe"
        location = blobs.put(probe, checksum_of(probe))
        # put only buffers for a packing backend; flush is what makes an
        # object exist, so that is what decides whether there is one to
        # clean up
        blobs.flush()
        uploaded = location
        read = blobs.get(location)

        if read == body:
            console.print(f"  [green]round trip correct[/green] — {len(read):,} bytes")
        else:
            ok = False
            console.print(
                f"  [red]wrong bytes back[/red] — asked for {len(body):,} at "
                f"offset {location.offset}, got {len(read):,}."
            )
            if endpoint:
                console.print(
                    "  [red]Object storage that ignores Range returns the start "
                    "of the shard for every sample. Nothing downstream would "
                    "notice.[/red]"
                )
    except Exception as e:
        ok = False
        console.print(f"  [red]failed[/red] — {type(e).__name__}: {e}")
    finally:
        probe.unlink(missing_ok=True)
        if endpoint and uploaded is not None:
            try:
                blobs.client.delete_object(
                    Bucket=config.s3_bucket, Key=uploaded.container
                )
            except Exception:
                console.print(
                    f"  [dim]left a probe object behind at {uploaded.container}[/dim]"
                )

    raise typer.Exit(0 if ok else 1)
