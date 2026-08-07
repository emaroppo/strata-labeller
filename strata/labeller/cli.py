import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from strata.modelling import Model

from .config import Settings
from .project import PROJECT_ENV_VAR, PROJECTS_DIR, Project, ProjectError

app = typer.Typer(name="auto-labeller")
console = Console()

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


def _resolve_checkpoint(
    model: Model,
    project: Project,
    checkpoint: Path | None = None,
    fresh: bool = False,
) -> Path | None:
    """Load a checkpoint into model. Returns the path loaded, or None."""
    if fresh:
        return None
    if checkpoint is not None:
        model.load(checkpoint)
        return checkpoint
    latest = project.latest_checkpoint()
    if latest is not None:
        model.load(latest)
    return latest


def _require_checkpoint(model: Model, project: Project, checkpoint: Path | None) -> Path:
    ckpt = _resolve_checkpoint(model, project, checkpoint)
    if not ckpt:
        console.print("[red]No checkpoint found. Run 'train' first or pass --checkpoint.[/red]")
        raise typer.Exit(1)
    console.print(f"Using checkpoint: {ckpt}")
    return ckpt


def _push_with_progress(client, project_id, predictions, task_id_map, **kwargs) -> int:
    total = sum(1 for p in predictions if p.path in task_id_map)
    with Progress(
        TextColumn("[bold cyan]Pushing"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("push", total=total)
        return client.push_predictions(
            project_id,
            predictions,
            task_id_map,
            on_progress=lambda n: progress.advance(bar, n),
            **kwargs,
        )


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
    from .dataset import load_dataset
    from .project import list_projects

    found = list_projects()
    if not found:
        console.print(
            "[yellow]No projects yet. Create one with 'auto-labeller new <name>'.[/yellow]"
        )
        return

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
        total = labeled = 0
        if project.dataset_path.exists():
            dataset = load_dataset(project.dataset_path, project.schema)
            total = len(dataset)
            labeled = sum(1 for s in dataset if s.is_labeled)
        table.add_row(
            project.name,
            project.schema.type,
            ", ".join(project.schema.classes) or "-",
            str(total),
            str(labeled),
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
    from .dataset import get_classes, load_dataset
    from .label_config import LabelConfigError

    project = _load_project(project_path)

    dataset = []
    if project.dataset_path.exists():
        dataset = load_dataset(project.dataset_path, project.schema)

    try:
        classes = project.add_classes(names, known=get_classes(dataset, project.schema))
    except ProjectError as e:
        _error(str(e))
        raise typer.Exit(1) from None
    console.print(f"[green]Added {', '.join(names)}[/green] — classes: {', '.join(classes)}")

    if push and project.label_studio.project_id is not None:
        settings = Settings.load(config_path)
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

    labeled = sum(1 for s in dataset if s.is_labeled)
    skipped = sum(1 for s in dataset if s.skipped)
    if labeled:
        console.print(
            f"[dim]{labeled} samples were labeled before this class existed.[/dim]"
        )
    if skipped:
        console.print(
            f"[dim]{skipped} skipped samples may contain it — "
            f"'auto-labeller unskip' returns them to the review queue.[/dim]"
        )


@class_app.command("list")
def class_list(
    project_path: Path | None = ProjectOption,
) -> None:
    """List the project's classes with how many samples carry each."""
    from .dataset import load_dataset

    project = _load_project(project_path)
    dataset = (
        load_dataset(project.dataset_path, project.schema)
        if project.dataset_path.exists()
        else []
    )

    schema = project.schema
    counts: dict[str, int] = {c: 0 for c in schema.classes}
    for sample in dataset:
        for name in schema.classes_in_use([sample.results]):
            counts[name] = counts.get(name, 0) + 1

    table = Table(title=f"Classes — {project.name}")
    table.add_column("Class", style="cyan")
    table.add_column("Samples", justify="right", style="green")
    table.add_column("", style="yellow")
    for name, count in counts.items():
        undeclared = "not in the label config" if name not in schema.classes else ""
        table.add_row(name, str(count), undeclared)
    console.print(table)


@app.command()
def unskip(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
    limit: int | None = typer.Option(None, help="Return only the first N skipped samples"),
) -> None:
    """Return skipped samples to the review queue.

    Skipping is how you park an image whose content has no class yet, so
    after adding a class those samples are the best place to find examples
    of it. In Label Studio a skip is a cancelled annotation, which is
    deleted here so the task becomes reviewable again.
    """
    from .dataset import load_dataset, save_dataset

    project = _load_project(project_path)
    dataset = load_dataset(project.dataset_path, project.schema)
    skipped = [s for s in dataset if s.skipped]
    if not skipped:
        console.print("[yellow]No skipped samples.[/yellow]")
        raise typer.Exit(0)

    selected = skipped[:limit] if limit is not None else skipped

    if project.label_studio.project_id is not None:
        settings = Settings.load(config_path)
        client = _ls_client(settings, project, config_path)
        project_id = project.label_studio.project_id
        with console.status("Fetching task list from Label Studio..."):
            task_id_map = client.get_task_id_map(project_id)
        task_ids = [task_id_map[s.path] for s in selected if s.path in task_id_map]
        if task_ids:
            with console.status(f"Clearing skips on {len(task_ids)} tasks..."):
                client.delete_annotations(project_id, task_ids)
        console.print(f"Cleared the skip on {len(task_ids)} Label Studio tasks")

    unskipped = {s.path for s in selected}
    for sample in dataset:
        if sample.path in unskipped:
            sample.skipped = False
    save_dataset(dataset, project.dataset_path)

    console.print(
        f"[green]Returned {len(selected)} samples to the unlabeled pool[/green] "
        f"({len(skipped) - len(selected)} still skipped)"
    )
    console.print("Run 'auto-labeller push' to queue them with fresh predictions.")


def _catalog_for(settings, config_path: Path):
    """The catalog this host holds, or an exit with something actionable."""
    from strata.catalog import Catalog

    root = Path(settings.catalog.root)
    if not (root / "catalog.db").exists():
        _error(
            f"No catalog at {root}. Run 'auto-labeller to-catalog' first, or set "
            f"[catalog] root in {config_path}."
        )
        raise typer.Exit(1)
    return Catalog.local(root), root


def _label_set_for(catalog, project: Project):
    from strata.catalog import CatalogError

    try:
        return catalog.label_set(project.label_set_name)
    except CatalogError:
        _error(
            f"No label set named '{project.label_set_name}' in the catalog. "
            f"Run 'auto-labeller to-catalog', or set [catalog] label_set."
        )
        raise typer.Exit(1) from None


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
    samples = catalog.unlabelled(label_set_id) + catalog.labelled(label_set_id)
    if limit is not None:
        samples = samples[:limit]

    client = _ls_client(settings, project, config_path)
    ls_project_id = client.create_project(project.name)
    client.setup_local_storage(
        ls_project_id, path=f"/label-studio/data/{settings.catalog.blobs_prefix}"
    )

    tasks, _ = tasks_to_push(
        samples, catalog, label_set_id, schema, settings.catalog.blobs_prefix, {}
    )
    mapping = client.import_catalog_tasks(ls_project_id, tasks)
    save_task_map(project, ls_project_id, mapping)
    project.save_ls_project_id(ls_project_id)

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
) -> None:
    """Scan the project's data root for new files and add them to the dataset.

    Which files count comes from the project's media type, so a text
    project picks up documents where an image project picks up pictures.
    Label Studio tasks are created on demand by `push`, so new files only
    need to be registered here.
    """
    from .dataset import Sample, load_dataset, save_dataset

    project = _load_project(project_path)
    data_dir = project.data_dir
    media = project.schema.media

    if not data_dir.exists():
        console.print(f"[red]Data root does not exist: {data_dir}[/red]")
        raise typer.Exit(1)

    # Load existing dataset to find already-tracked paths
    dataset: list[Sample] = []
    if project.dataset_path.exists():
        dataset = load_dataset(project.dataset_path, project.schema)
    existing_paths = {s.path for s in dataset}

    # Sample paths are relative to the data root
    new_samples: list[Sample] = []
    for path in sorted(data_dir.rglob("*")):
        if media.matches(path.name):
            rel = str(path.relative_to(data_dir))
            if rel not in existing_paths:
                new_samples.append(Sample(path=rel))

    if not new_samples:
        console.print(
            f"[yellow]No new {media.name} files found "
            f"({', '.join(sorted(media.extensions))}).[/yellow]"
        )
        raise typer.Exit(0)

    updated = dataset + new_samples
    save_dataset(updated, project.dataset_path)
    console.print(
        f"[green]Added {len(new_samples)} new {media.name} files "
        f"({len(updated)} total samples)[/green]"
    )


@app.command()
def train(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
    fresh: bool = typer.Option(
        False, "--fresh/--no-fresh", help="Cold start, ignoring the previous run"
    ),
    val_ratio: float = typer.Option(0.2, help="Share of samples to hold out"),
    legacy: bool = typer.Option(
        False,
        "--legacy",
        help="Train from dataset.json instead of the catalog (removed once the cutover settles)",
    ),
    round_num: int | None = typer.Option(None, help="--legacy only: round number"),
    checkpoint: Path | None = typer.Option(None, help="--legacy only: checkpoint to continue from"),
) -> None:
    """Train on the project's labelled data.

    A round freezes a dataset version in the catalog, materialises it, and
    trains from that directory — so the run resolves back to the exact
    samples behind it, and validation membership is inherited rather than
    recomputed.
    """
    project = _load_project(project_path)

    if legacy:
        _train_from_dataset_json(project, round_num, checkpoint, fresh)
        return

    from strata.catalog import Catalog

    from .round import RoundError, describe, run_round

    settings = Settings.load(config_path)
    catalog_root = Path(settings.catalog.root)
    if not (catalog_root / "catalog.db").exists():
        console.print(
            f"[red]No catalog at {catalog_root}.[/red] Run 'auto-labeller to-catalog' "
            f"first, or set [catalog] root in {config_path}."
        )
        raise typer.Exit(1)

    console.print("Training...")
    try:
        result = run_round(project, Catalog.local(catalog_root), fresh=fresh, val_ratio=val_ratio)
    except RoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    for line in describe(result):
        console.print(line)
    console.print(f"  checkpoint: {result.run.checkpoint}")


def _train_from_dataset_json(
    project: Project, round_num: int | None, checkpoint: Path | None, fresh: bool
) -> None:
    """The pre-catalog round, kept for one cutover.

    Reads dataset.json and writes rounds/<n>/metadata.json. It cannot see
    anything labelled since the last `to-catalog`, and its split is
    recomputed each time rather than inherited.
    """
    from .train import run_training

    console.print("[yellow]--legacy: training from dataset.json, not the catalog.[/yellow]")
    model = project.load_model()
    ckpt = _resolve_checkpoint(model, project, checkpoint, fresh)
    if ckpt:
        console.print(f"Loaded checkpoint: {ckpt}")
    elif fresh:
        console.print("Training from scratch.")

    meta = run_training(model, project, round_num)
    console.print(f"[green]Round {meta['round']} complete[/green]")
    console.print(
        f"  Train: {meta['num_train']}, Val: {meta['num_val']}, "
        f"Unlabeled: {meta['num_unlabeled']}"
    )
    console.print(f"  Classes: {', '.join(meta['classes'])}")
    for k, v in (meta["metrics"] or {}).items():
        console.print(f"  {k}: {v}")


@app.command()
def predict(
    project_path: Path | None = ProjectOption,
    unlabeled_only: bool = typer.Option(True, help="Only predict on unlabeled samples"),
    checkpoint: Path | None = typer.Option(None, help="Checkpoint to use (default: latest)"),
    output: Path | None = typer.Option(None, help="Save predictions to JSON"),
) -> None:
    """Run model predictions on the dataset."""
    from .dataset import Sample, load_dataset, save_dataset, split_labeled_unlabeled
    from .predict import run_predictions

    project = _load_project(project_path)
    model = project.load_model()
    _require_checkpoint(model, project, checkpoint)

    dataset = load_dataset(project.dataset_path, project.schema)
    samples = (
        split_labeled_unlabeled(dataset)[1]
        if unlabeled_only
        else [s for s in dataset if not s.skipped]
    )

    if not samples:
        console.print("[yellow]No samples to predict on.[/yellow]")
        raise typer.Exit(0)

    console.print(f"Predicting on {len(samples)} samples...")
    predictions = run_predictions(model, samples, project)

    schema = project.schema
    for pred in predictions:
        summary = ", ".join(schema.classes_in_use([pred.results])) or "(nothing)"
        console.print(f"  {pred.path}: {summary} (score {pred.score:.2f})")

    if output:
        pred_samples = [
            Sample(path=p.path, results=p.results, annotated=False) for p in predictions
        ]
        save_dataset(pred_samples, output)
        console.print(f"[green]Saved predictions to {output}[/green]")


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

    from .active_learning import least_confident
    from .adapter import prediction_to_results
    from .sync import load_task_map, rebuild_task_map, save_task_map, tasks_to_push

    project = _load_project(project_path)
    settings = Settings.load(config_path)
    catalog, _ = _catalog_for(settings, config_path)
    label_set_id, _ = _label_set_for(catalog, project)
    prefix = settings.catalog.blobs_prefix
    schema = project.schema

    try:
        ls_project_id = project.require_ls_project_id()
    except ProjectError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    client = _ls_client(settings, project, config_path)
    task_map = load_task_map(project, ls_project_id)
    if rebuild_map or not task_map:
        with console.status("Listing tasks in Label Studio..."):
            task_map, unrecognised = rebuild_task_map(
                client.list_tasks(ls_project_id), catalog, prefix, schema.data_key
            )
        if unrecognised:
            console.print(
                f"[yellow]{len(unrecognised)} task(s) point at nothing this catalog "
                f"knows — from before the cutover, or since removed.[/yellow]"
            )
        save_task_map(project, ls_project_id, task_map)

    pool = catalog.unlabelled(label_set_id)
    if not pool:
        console.print("[yellow]Nothing is waiting for review.[/yellow]")
        return

    ranked, scored = pool, {}
    store = RunStore.local(project.runs_dir)
    run = store.get(run_id) if run_id else store.latest(project.dataset_name)

    if predictions and run is not None and run.checkpoint:
        with console.status(f"Predicting with run {run.id}..."):
            made = run_predict(
                PredictRequest(
                    run_id=run.id,
                    paths=[catalog.blobs.path_for(s.location) for s in pool],
                ),
                store,
            )
        by_sample = dict(zip(pool, made, strict=True))
        # Least confident first: what the model committed to least is what a
        # human settles fastest
        ranked = sorted(pool, key=lambda s: least_confident(by_sample[s].value), reverse=True)
        scored = {s.id: by_sample[s].value for s in pool}
    elif predictions:
        console.print("[yellow]No run with a checkpoint yet; pushing without predictions.[/yellow]")

    if limit is not None:
        ranked = ranked[:limit]

    tasks, report = tasks_to_push(
        ranked, catalog, label_set_id, schema, prefix, task_map
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
            (s.id, prediction_to_results(scored[s.id], schema), scored[s.id].confidences[0]
             if scored[s.id].confidences else 0.0)
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
    dataset_json: bool = typer.Option(
        True,
        "--dataset-json/--no-dataset-json",
        help="Also write dataset.json, so `train --legacy` keeps working",
    ),
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
        ls_project_id = project.require_ls_project_id()
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
        settings.catalog.blobs_prefix,
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

    if dataset_json:
        _mirror_to_dataset_json(project, catalog, label_set_id)


def _mirror_to_dataset_json(project: Project, catalog, label_set_id: int) -> None:
    """Keep dataset.json in step, so `train --legacy` remains a real fallback.

    Written from the catalog rather than merged with what is there: the
    catalog is the store now, and reconciling two of them is exactly what
    this stops being worth doing.
    """
    from .dataset import Sample, save_dataset

    schema = project.schema
    samples = []
    for row in catalog.labelled(label_set_id):
        value = catalog.annotation_of(row.id, label_set_id)
        samples.append(
            Sample(
                path=row.location.container,
                results=schema.encode_target(list(value.values)) if value else [],
                annotated=True,
            )
        )
    save_dataset(samples, project.dataset_path)
    console.print(f"  mirrored {len(samples)} labelled sample(s) to {project.dataset_path}")


@app.command()
def serve(
    project_path: Path | None = ProjectOption,
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(9090, help="Port"),
) -> None:
    """Start the ML backend server for Label Studio live predictions."""
    import uvicorn

    project = _load_project(project_path)
    # uvicorn imports the app in a worker process, so the project travels
    # through the environment rather than as an argument
    os.environ[PROJECT_ENV_VAR] = str(project.root.resolve())

    console.print(f"Starting ML backend for '{project.name}' on {host}:{port}")
    console.print("Add this URL as an ML backend in Label Studio:")
    console.print(f"  http://host.docker.internal:{port}")
    uvicorn.run(
        "strata.labeller.ls_backend:app",
        host=host,
        port=port,
        reload=True,
    )


@app.command()
def report(
    project_path: Path | None = ProjectOption,
    round_num: int | None = typer.Option(None, help="Round number (latest if omitted)"),
) -> None:
    """Show a round summary report."""
    project = _load_project(project_path)
    rounds_dir = project.rounds_dir

    if not rounds_dir.exists():
        console.print("[red]No rounds found. Run 'train' first.[/red]")
        raise typer.Exit(1)

    round_dirs = sorted(
        d for d in rounds_dir.iterdir() if d.is_dir() and d.name.startswith("round_")
    )
    if not round_dirs:
        console.print("[red]No rounds found.[/red]")
        raise typer.Exit(1)

    if round_num is not None:
        round_dir = rounds_dir / f"round_{round_num:03d}"
    else:
        round_dir = round_dirs[-1]

    meta_path = round_dir / "metadata.json"
    if not meta_path.exists():
        console.print(f"[red]No metadata found for {round_dir.name}[/red]")
        raise typer.Exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    table = Table(title=f"Round {meta['round']} Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Timestamp", meta["timestamp"])
    table.add_row("Training samples", str(meta["num_train"]))
    table.add_row("Validation samples", str(meta["num_val"]))
    table.add_row("Unlabeled samples", str(meta["num_unlabeled"]))
    if meta.get("num_skipped") is not None:
        table.add_row("Skipped samples", str(meta["num_skipped"]))
    table.add_row("Classes", ", ".join(meta["classes"]))
    table.add_row("Checkpoint", meta["checkpoint"])

    if meta.get("metrics"):
        for k, v in meta["metrics"].items():
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))

    console.print(table)

    if len(round_dirs) > 1 and round_dir == round_dirs[-1]:
        prev_meta_path = round_dirs[-2] / "metadata.json"
        if prev_meta_path.exists():
            with open(prev_meta_path) as f:
                prev_meta = json.load(f)

            console.print(f"\n[bold]Delta vs Round {prev_meta['round']}:[/bold]")
            delta_train = meta["num_train"] - prev_meta["num_train"]
            console.print(f"  Training samples: {'+' if delta_train >= 0 else ''}{delta_train}")

            if meta.get("metrics") and prev_meta.get("metrics"):
                for k in meta["metrics"]:
                    if k in prev_meta["metrics"]:
                        curr = meta["metrics"][k]
                        prev = prev_meta["metrics"][k]
                        if isinstance(curr, (int, float)) and isinstance(prev, (int, float)):
                            delta = curr - prev
                            console.print(f"  {k}: {'+' if delta >= 0 else ''}{delta:.4f}")


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
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

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
    catalog_root: Path = typer.Option(
        Path("catalog"), "--catalog", help="The catalog to read"
    ),
) -> None:
    """What is in a catalog: samples, label sets, and the class breakdown.

    The per-class counts come from the annotation_class index rather than
    from scanning stored values, so they are also the check that indexing
    did its job.
    """
    from sqlalchemy import func, select

    from strata.catalog import Catalog
    from strata.catalog import tables as t

    if not (catalog_root / "catalog.db").exists():
        console.print(f"[red]No catalog at {catalog_root}[/red]")
        raise typer.Exit(1)

    catalog = Catalog.local(catalog_root)
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
        label_sets = conn.execute(select(t.label_set.c.id, t.label_set.c.name)).all()

    console.print(f"[bold]{catalog_root}[/bold]: {total} sample(s)")
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

    if not label_sets:
        console.print("[yellow]No label sets yet.[/yellow]")
        return

    for label_set_id, name in label_sets:
        _, schema = catalog.label_set(name)
        labelled = catalog.labelled(label_set_id)
        queue = catalog.unlabelled(label_set_id)
        console.print(
            f"\n[bold]{name}[/bold] — {schema.task}, "
            f"{'multi' if schema.multiple else 'single'}-choice"
        )
        console.print(f"  {len(labelled)} annotated, {len(queue)} awaiting review")

        table = Table("Class", "Samples", box=None, pad_edge=False)
        for class_name in schema.classes:
            table.add_row(class_name, str(len(catalog.with_class(label_set_id, class_name))))
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
