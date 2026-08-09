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


def _warn_undeclared(project: Project, samples: list) -> list[str]:
    """Warn about classes present in the data but missing from the schema.

    A class added in the Label Studio UI only trains as nothing: the model's
    head is built from the declared list, so targets outside it are dropped
    without a word.
    """
    from .dataset import get_classes

    schema = project.schema
    declared = set(schema.classes)
    if not declared:
        return []
    undeclared = [c for c in get_classes(samples, schema) if c not in declared]
    if undeclared:
        console.print(
            f"[yellow]Not in project.toml: {', '.join(undeclared)} — "
            f"these labels are ignored during training until declared:[/yellow]"
        )
        console.print(f"  auto-labeller class add {' '.join(undeclared)}")
    return undeclared


def _blobs_for(settings, root: Path):
    """Where this host reads and writes sample bytes.

    Files under the catalog root when nothing else is configured, which is
    what keeps a checkout working. An endpoint means tar shards in a bucket,
    which is what lets the machine that trains and the machine that labels
    read the same bytes without either owning them.
    """
    from strata.catalog import LocalBackend

    if not settings.catalog.s3_endpoint:
        return LocalBackend(root / "blobs")

    import boto3
    from botocore.config import Config

    from strata.catalog.s3 import S3Backend

    client = boto3.client(
        "s3",
        endpoint_url=settings.catalog.s3_endpoint,
        aws_access_key_id=settings.catalog.s3_access_key or None,
        aws_secret_access_key=settings.catalog.s3_secret_key or None,
        region_name=settings.catalog.s3_region,
        # Anything that is not AWS serves buckets as a path rather than as a
        # subdomain, and the default guesses the other way
        config=Config(s3={"addressing_style": "path"}),
    )
    return S3Backend(client, bucket=settings.catalog.s3_bucket)


def _addressing(settings):
    """How this host writes and reads task image URLs.

    One place, because the two directions have to agree: pushing HTTP URLs
    while reading local ones would orphan every task, and the symptom is an
    empty export rather than an error.
    """
    from .adapter import AdapterError, Addressing

    try:
        return Addressing(
            prefix=settings.catalog.blobs_prefix,
            base_url=settings.catalog.serve_url,
            secret=settings.catalog.blob_secret,
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
                f"this host. Push without --predictions, or run this where the "
                f"blobs are — fetching a whole review pool out of the bucket "
                f"to rank it is not something to do by accident."
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


def _catalog_for(settings, config_path: Path, create: bool = False):
    """The catalog this host holds, or an exit with something actionable.

    ``create`` for the commands that put data in: refusing to make one would
    leave no way to make the first, and the advice would be circular.
    """
    from strata.catalog import Catalog

    root = Path(settings.catalog.root)
    blobs = _blobs_for(settings, root)
    if settings.catalog.url:
        # A shared index: nothing local to check for, and create_all is
        # harmless against one that already exists
        return Catalog.connect(settings.catalog.url, blobs), root

    if not create and not (root / "catalog.db").exists():
        _error(
            f"No catalog at {root}. Run 'auto-labeller ingest' or "
            f"'auto-labeller to-catalog' to make one, or point [catalog] root "
            f"in {config_path} at an existing one."
        )
        raise typer.Exit(1)
    root.mkdir(parents=True, exist_ok=True)
    return Catalog.connect(f"sqlite:///{root / 'catalog.db'}", blobs), root


def _catalog_if_any(settings):
    """The configured catalog, or None when this host has none yet.

    For the paths that decorate output with counts, where absence is not an
    error: a project that has never ingested is still a project. Distinct
    from :func:`_catalog_for`, which exits — these callers must not.

    What it is not is a shortcut to the local one. Checking for a
    ``catalog.db`` and opening it regardless of configuration is how five
    commands ended up reading a stale SQLite index after the catalog moved
    to Postgres, reporting counts from it as though they were current.
    """
    from strata.catalog import Catalog

    root = Path(settings.catalog.root)
    if settings.catalog.url:
        return Catalog.connect(settings.catalog.url, _blobs_for(settings, root))
    if (root / "catalog.db").exists():
        return Catalog.connect(
            f"sqlite:///{root / 'catalog.db'}", _blobs_for(settings, root)
        )
    return None


def _label_set_for(catalog, project: Project):
    from strata.catalog import CatalogError

    try:
        return catalog.label_set(project.label_set_name)
    except CatalogError:
        _error(
            f"No label set named '{project.label_set_name}' in the catalog. "
            f"Run 'auto-labeller ingest' or 'auto-labeller to-catalog', or set "
            f"[catalog] label_set."
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
    table.add_column("Files", style="magenta")
    table.add_column("Annotations", style="green")
    for name in schemas.available_templates():
        if name == schemas.CUSTOM_TEMPLATE:
            table.add_row(name, "-", "whatever label_config.xml declares")
            continue
        spec = schemas.TEMPLATES[name]
        extensions = ", ".join(f".{e}" for e in sorted(spec.media.extensions))
        table.add_row(name, extensions, spec.control_tag)
    console.print(table)


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
    # one — a project that has never ingested is still a project
    settings = Settings.load()
    catalog = _catalog_if_any(settings)

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
    settings = Settings.load(config_path)

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

    catalog = _catalog_if_any(settings)
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

    catalog = _catalog_if_any(settings)
    if catalog is None:
        return
    try:
        label_set_id, schema = catalog.label_set(project.label_set_name)
    except CatalogError:
        # No label set yet: to-catalog creates it from project.toml, so the
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
    settings = Settings.load(config_path)
    catalog, _ = _catalog_for(settings, config_path)
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
    from .sync import load_task_map

    project = _load_project(project_path)
    settings = Settings.load(config_path)
    catalog, _ = _catalog_for(settings, config_path)
    label_set_id, _ = _label_set_for(catalog, project)

    skipped = catalog.skipped(label_set_id, project.collections)
    if not skipped:
        console.print("[yellow]Nothing is skipped.[/yellow]")
        return

    selected = skipped[:limit] if limit is not None else skipped

    ls_project_id = project.label_studio.project_id
    if ls_project_id is not None:
        task_map = load_task_map(project, ls_project_id)
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
    settings = Settings.load(config_path)
    catalog, _ = _catalog_for(settings, config_path)
    label_set_id, label_schema = _label_set_for(catalog, project)

    if not label_schema.classes:
        _error("The label set declares no classes; add some before labelling.")
        raise typer.Exit(1)

    schema = project.schema
    # Answered first: with a limit, the point is to carry what is already
    # known rather than to fill the project with unreviewed samples
    samples = catalog.labelled(label_set_id, project.collections) + catalog.unlabelled(
        label_set_id, project.collections
    )
    if limit is not None:
        samples = samples[:limit]

    client = _ls_client(settings, project, config_path)
    ls_project_id = client.create_project(project.name)
    if not settings.catalog.serve_url:
        # Only when Label Studio is the one reading files. Once tasks carry
        # signed URLs to the serving API, a local storage connection points
        # at a mount this deployment no longer has, and configuring one
        # would suggest the mount still matters.
        client.setup_local_storage(
            ls_project_id, path=f"/label-studio/data/{settings.catalog.blobs_prefix}"
        )

    tasks, _ = tasks_to_push(
        samples, catalog, label_set_id, schema, _addressing(settings), {}
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
    save_task_map(project, ls_project_id, mapping)
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

    Which files count comes from the project's media type, so a text project
    picks up documents where an image project picks up pictures. Grouping
    comes from [data] kind: video frames are grouped by the folder they sit
    in, so near-duplicates cannot straddle a train/val split.

    Registering is not queueing. The catalog holds the whole pool and `push`
    sends only what you are about to review, so there is no cost to
    cataloguing everything.
    """
    from strata.catalog import CatalogError

    from .to_catalog import group_id_for, schema_for

    project = _load_project(project_path)
    settings = Settings.load(config_path)
    data_dir = project.data_dir
    media = project.schema.media

    if not data_dir.exists():
        _error(f"Data root does not exist: {data_dir}")
        raise typer.Exit(1)

    catalog, catalog_root = _catalog_for(settings, config_path, create=True)
    try:
        label_set_id, _ = catalog.label_set(project.label_set_name)
    except CatalogError:
        label_set_id = catalog.create_label_set(project.label_set_name, schema_for(project))
        console.print(f"Created label set '{project.label_set_name}'")

    found = [p for p in sorted(data_dir.rglob("*")) if media.matches(p.name)]
    if not found:
        console.print(
            f"[yellow]No {media.name} files under {data_dir} "
            f"({', '.join(sorted(media.extensions))}).[/yellow]"
        )
        return

    where = project.collections
    before = len(catalog.unlabelled(label_set_id, where)) + len(
        catalog.labelled(label_set_id, where)
    )
    subtype = "frames" if project.data.kind == "frames" else "plain"

    # Grouped, because a group is one transaction and one group_id. Ungrouped
    # files share a bucket, so a plain image project is a handful of batches
    # rather than one per file.
    by_group: dict[str | None, list[Path]] = {}
    for path in found:
        relative = str(path.relative_to(data_dir))
        by_group.setdefault(group_id_for(project, relative), []).append(path)

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
                catalog.ingest(
                    chunk,
                    media=media.name,
                    subtype=subtype,
                    group_id=group_id,
                    metadata_for=lambda p: {"source_path": sources[p]},
                    collections=project.collections,
                    on_sample=lambda _p: progress.advance(bar),
                )

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

    settings = Settings.load(config_path)
    catalog, catalog_root = _catalog_for(settings, config_path)

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


def _run_for_push(settings, project, store, run_id, remote: bool):
    """Which run scores this push, in the numbering of whoever will score it.

    Run ids belong to the store that issued them. Asking a modelling host to
    predict with a local run id names a different model there, or none —
    silently, since both stores number from one.
    """
    if not remote:
        run = store.get(run_id) if run_id else store.latest(project.dataset_name)
        if run is None or not run.checkpoint:
            return None
        return {"id": run.id}

    from .remote import RemoteError, Trainer

    trainer = Trainer(settings.modelling.url, settings.modelling.token)
    try:
        # Asked for by id or not, the host is the one that knows. Checking
        # now costs one request; not checking costs a pool fetched and
        # scored before anything notices.
        found = trainer.run(run_id) if run_id else trainer.latest_run(project.dataset_name)
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
    return {"id": found["run"]["id"]}


def _remote_predictions(settings, run_id: int, checksums: list[str]) -> dict:
    """Score a review pool on the host that has the GPU and the blobs.

    The same job machinery as a round, for the same reason: this is minutes
    of work over tens of thousands of samples, and a laptop that closes
    should not take it with it.
    """
    from strata.labels import ChoicesPrediction

    from .remote import RemoteError, Trainer

    if not settings.modelling.token:
        _error(
            "No token for the modelling host. Set $STRATA_MODELLING_TOKEN to "
            "the same value it was started with."
        )
        raise typer.Exit(1)

    trainer = Trainer(settings.modelling.url, settings.modelling.token)
    try:
        job = trainer.predict(run_id, checksums)
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
    return {
        checksum: ChoicesPrediction.model_validate(value)
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


def _remote_round(project, catalog, settings, fresh: bool, val_ratio: float) -> None:
    """Freeze a dataset here, and have another host train on it.

    The split is where the knowledge is. Which samples make a dataset is the
    project's business — its collections, its label set, its val ratio — and
    the catalog is reachable from both machines. Everything after that needs
    a GPU and the checkpoints, and both live there.
    """
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

    dataset_id = catalog.create_dataset(
        project.dataset_name, label_set_id, collections=project.collections, val_ratio=val_ratio
    )
    name, version = catalog.dataset_named(dataset_id)
    console.print(f"Dataset {name} v{version} → {settings.modelling.url}")

    trainer = Trainer(settings.modelling.url, settings.modelling.token)
    try:
        job = trainer.submit(
            dataset_id, project.model_ref, project.model.params_for(fresh), fresh
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
    run_id: int | None = typer.Option(None, help="Predict with this run (default: latest)"),
    predictions: bool = typer.Option(
        True, "--predictions/--no-predictions", help="Attach pre-annotations"
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Replace existing predictions rather than adding to them"
    ),
    rebuild_map: bool = typer.Option(
        False, "--rebuild-map", help="Re-list tasks instead of trusting the local cache"
    ),
) -> None:
    """Send unreviewed samples to Label Studio, least confident first.

    Predictions come from a recorded run, so what a reviewer sees is tied to
    a checkpoint that resolves back to the data behind it.
    """
    from strata.modelling import PredictRequest, RunStore
    from strata.modelling import predict as run_predict

    from .active_learning import certainty, rank
    from .adapter import prediction_to_results
    from .predictions import PredictionCache
    from .sync import load_task_map, rebuild_task_map, save_task_map, tasks_to_push

    project = _load_project(project_path)
    settings = Settings.load(config_path)
    catalog, catalog_root = _catalog_for(settings, config_path)
    label_set_id, _ = _label_set_for(catalog, project)
    addressing = _addressing(settings)
    schema = project.schema

    try:
        ls_project_id = project.require_ls_project_id(settings.label_studio.url)
    except ProjectError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    client = _ls_client(settings, project, config_path)
    task_map = load_task_map(project, ls_project_id)
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
        save_task_map(project, ls_project_id, task_map)

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
    run = _run_for_push(settings, project, store, run_id, remote)

    if predictions and run is not None:
        checksums = [s.checksum for s in pool]
        if remote:
            # The host keeps its own cache, keyed on its own run ids — which
            # is the only place that key means anything.
            scores = _remote_predictions(settings, run["id"], checksums)
        else:
            from strata.modelling import PredictionCache

            cache = PredictionCache.local(project.runs_dir)
            scores = cache.get(run["id"], checksums)
            missing = [s for s in pool if s.checksum not in scores]
            if scores:
                console.print(
                    f"[dim]{len(scores):,} prediction(s) reused from run "
                    f"{run['id']}; {len(missing):,} to make[/dim]"
                )
            if missing:
                with console.status(f"Predicting with run {run['id']}..."):
                    fresh = run_predict(
                        PredictRequest(
                            run_id=run["id"], paths=_local_paths(catalog_root, missing)
                        ),
                        store,
                    )
                made = {s.checksum: p.value for s, p in zip(missing, fresh, strict=True)}
                cache.put(run["id"], made)
                scores.update(made)

        ranked = rank(pool, scores)
        scored = {s.id: scores[s.checksum] for s in ranked}
    elif predictions:
        console.print("[yellow]No run with a checkpoint yet; pushing without predictions.[/yellow]")

    if limit is not None:
        ranked = ranked[:limit]

    tasks, report = tasks_to_push(
        ranked, catalog, label_set_id, schema, addressing, task_map
    )
    created = client.import_catalog_tasks(ls_project_id, tasks)
    task_map.update(created)
    save_task_map(project, ls_project_id, task_map)

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
            model_version=f"run-{run.id}",
            replace_existing=refresh,
        )
        console.print(f"[green]{pushed} pre-annotation(s) attached from run {run.id}[/green]")


@app.command(name="export")
def export_annotations(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
) -> None:
    """Pull corrected annotations out of Label Studio into the catalog.

    The catalog is what remembers; Label Studio is where the answering
    happens. dataset.json is written alongside for as long as the legacy
    training path is worth keeping.
    """
    from .sync import pull_annotations

    project = _load_project(project_path)
    settings = Settings.load(config_path)
    catalog, _ = _catalog_for(settings, config_path)
    label_set_id, label_schema = _label_set_for(catalog, project)
    schema = project.schema

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
        _addressing(settings),
        label_schema.classes,
    )

    if report.undeclared:
        # The catalog validates against the label set, so this would fail
        # partway through rather than at the end
        _error(
            f"Label(s) nobody declared: {', '.join(sorted(report.undeclared))}. "
            f"Add them with 'auto-labeller class add', then export again."
        )
        raise typer.Exit(1)

    annotated, skipped = catalog.annotate_many(label_set_id, items, source="human")
    console.print(
        f"[green]{annotated} annotation(s) and {skipped} skip(s) into the catalog[/green]"
    )
    if report.unrecognised:
        console.print(
            f"[yellow]{len(report.unrecognised)} task(s) point at nothing this "
            f"catalog knows, and were left alone.[/yellow]"
        )


@app.command()
def report(
    project_path: Path | None = ProjectOption,
    metric: str = typer.Option("val_accuracy", help="Which metric to plot"),
    run_id: int | None = typer.Option(
        None, "--run", help="Detail one run instead of the history"
    ),
) -> None:
    """Show the training history, or one run in detail.

    Read from the run store rather than from round folders, so a metric
    across rounds is one query.

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

    if run_id is not None:
        run = store.get(run_id)
        if run is None:
            _error(f"No run with id {run_id}")
            raise typer.Exit(1)
        _print_run(store, run)
        return

    history = store.history(project.dataset_name, metric)
    if not history:
        _error(
            f"No run recorded {metric!r} for '{project.dataset_name}'. "
            f"Try --metric accuracy, or --run to inspect one."
        )
        raise typer.Exit(1)

    table = Table(title=f"{project.dataset_name} — {metric}")
    table.add_column("Run", justify="right")
    table.add_column("Dataset", justify="right")
    table.add_column(metric, justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("Lineage")

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
        delta = f"{value - seen[parent]:+.4f}" if comparable else ""
        lineage = f"from {parent}" if parent else "[yellow]unchained[/yellow]"
        shown = f"v{version}" if version is not None else "[dim]—[/dim]"
        table.add_row(str(run_num), shown, f"{value:.4f}", delta, lineage)
        seen[run_num] = value
        versions[run_num] = version
    console.print(table)


def _print_run(store, run) -> None:
    console.print(f"[bold]Run {run.id}[/bold] — {run.model} v{run.model_version}")
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
        console.print(f"  continues: {' -> '.join(str(r.id) for r in chain)}")
    else:
        console.print("  unchained — continues nothing in the store")

    if run.metrics:
        table = Table("Metric", "Value", box=None, pad_edge=False)
        for name, value in sorted(run.metrics.items()):
            table.add_row(name, f"{value:.4f}")
        console.print(table)


@app.command(name="to-catalog")
def to_catalog(
    project_path: Path | None = ProjectOption,
    catalog_root: Path = typer.Option(
        Path("catalog"), "--catalog", help="Where the catalog lives (created if absent)"
    ),
    label_set: str | None = typer.Option(
        None, "--label-set", help="Name for the label set (default: the project's name)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report without writing anything"),
) -> None:
    """Move this project's dataset.json into a catalog.

    One direction, and safe to repeat: samples are addressed by content and
    annotations are upserted, so a run that stopped halfway can just be run
    again. Nothing about the project is modified.
    """
    from strata.catalog import Catalog

    from .to_catalog import MigrationError, describe, migrate

    project = _load_project(project_path)
    if not project.dataset_path.exists():
        console.print(f"[red]No dataset at {project.dataset_path}[/red]")
        raise typer.Exit(1)

    catalog = None if dry_run else Catalog.local(catalog_root)
    try:
        if dry_run:
            report = migrate(project, catalog, label_set=label_set, dry_run=True)
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Migrating", total=None)

                def advance(done: int, total: int) -> None:
                    # Total is only known once the files have been resolved,
                    # so the bar starts indeterminate and settles
                    progress.update(task, completed=done, total=total)

                report = migrate(
                    project, catalog, label_set=label_set, on_progress=advance
                )
    except MigrationError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    if dry_run:
        console.print("[yellow]Dry run — nothing was written.[/yellow]")
    console.print(describe(report, catalog_root))


@app.command(name="catalog-stats")
def catalog_stats(
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

    settings = Settings.load(config_path)
    catalog, catalog_root = _catalog_for(settings, config_path)
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

    where = _redacted(settings.catalog.url) if settings.catalog.url else catalog_root
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
        console.print(
            f"\n[bold]{name}[/bold] — {schema.task}, "
            f"{'multi' if schema.multiple else 'single'}-choice"
        )
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


@app.command(name="catalog-copy")
def catalog_copy(
    to_url: str = typer.Option(..., "--to", help="Index URL to copy into"),
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

    settings = Settings.load(config_path)
    source, root = _catalog_for(settings, config_path)
    target = Catalog.connect(to_url, _blobs_for(settings, root))

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
    settings = Settings.load(config_path)
    catalog, _ = _catalog_for(settings, config_path)
    addressing = _addressing(settings)
    schema = project.schema

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

    settings = Settings.load(config_path)
    if not settings.catalog.s3_endpoint:
        _error(
            "No object storage configured. Set STRATA_S3_ENDPOINT and "
            "STRATA_S3_BUCKET (and the credentials) — this packs blobs into a "
            "bucket, so there is nowhere to put them otherwise."
        )
        raise typer.Exit(1)

    catalog, root = _catalog_for(settings, config_path)
    target = _blobs_for(settings, root)
    # Before any put, since the size is read when a shard is opened. Larger
    # shards mean fewer objects and fewer requests; the cost is memory, as a
    # shard is buffered whole and then read into one bytes object to upload.
    target.shard_bytes = shard_mb * 1024 * 1024
    # Explicitly the local one: _blobs_for answers with the bucket once an
    # endpoint is set, and that is the destination, not the source.
    source = LocalBackend(root / "blobs")

    console.print(f"[bold]from[/bold]  {source.root}")
    console.print(
        f"[bold]to[/bold]    {settings.catalog.s3_endpoint} "
        f"bucket={settings.catalog.s3_bucket} shards={shard_mb} MB\n"
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


@app.command(name="catalog-probe")
def catalog_probe(
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

    settings = Settings.load(config_path)
    ok = True

    # -- the index -----------------------------------------------------
    where = (
        _redacted(settings.catalog.url)
        if settings.catalog.url
        else f"sqlite under {settings.catalog.root}"
    )
    console.print(f"[bold]index[/bold]  {where}")
    try:
        catalog, root = _catalog_for(settings, config_path, create=True)
        with catalog.engine.connect() as conn:
            samples = conn.execute(select(func.count()).select_from(t.sample)).scalar()
        console.print(f"  [green]reachable[/green] — {samples:,} sample(s)")
    except Exception as e:
        console.print(f"  [red]unreachable[/red] — {type(e).__name__}: {e}")
        raise typer.Exit(1) from None

    # -- the blobs -----------------------------------------------------
    endpoint = settings.catalog.s3_endpoint
    console.print(
        "\n[bold]blobs[/bold]  "
        + (f"{endpoint} bucket={settings.catalog.s3_bucket}" if endpoint
           else f"files under {root / 'blobs'}")
    )

    body = bytes(range(256)) * 64
    probe = root / f".probe-{uuid.uuid4().hex[:8]}"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_bytes(body)
    uploaded = None
    try:
        blobs = _blobs_for(settings, root)
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
                    Bucket=settings.catalog.s3_bucket, Key=uploaded.container
                )
            except Exception:
                console.print(
                    f"  [dim]left a probe object behind at {uploaded.container}[/dim]"
                )

    raise typer.Exit(0 if ok else 1)
