import json
import os
import random
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

from .config import Settings
from .model import BaseModel
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
    model: BaseModel,
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


def _require_checkpoint(model: BaseModel, project: Project, checkpoint: Path | None) -> Path:
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


@app.command()
def init(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
) -> None:
    """Create the Label Studio project and import the dataset's tasks."""
    from .dataset import get_classes, load_dataset

    project = _load_project(project_path)
    settings = Settings.load(config_path)
    schema = project.schema
    dataset = load_dataset(project.dataset_path, schema)
    classes = schema.classes or get_classes(dataset, schema)

    if not classes:
        _error("No classes: set [label_config] classes in project.toml.")
        raise typer.Exit(1)

    client = _ls_client(settings, project, config_path)
    project_id = client.create_project(project.name)
    client.setup_local_storage(project_id)
    client.import_tasks(project_id, dataset)
    project.save_ls_project_id(project_id)

    console.print(
        f"[green]Created Label Studio project '{project.name}' (ID: {project_id}) "
        f"with {len(dataset)} tasks[/green]"
    )
    console.print(f"Classes: {', '.join(classes)}")


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
    round_num: int | None = typer.Option(
        None, help="Round number (auto-detected if omitted)"
    ),
    checkpoint: Path | None = typer.Option(
        None, help="Checkpoint to continue from (default: latest)"
    ),
    fresh: bool = typer.Option(
        False, "--fresh/--no-fresh", help="Train from scratch, ignoring existing checkpoints"
    ),
) -> None:
    """Train the project's model on its labeled data."""
    from .train import run_training

    project = _load_project(project_path)
    model = project.load_model()

    ckpt = _resolve_checkpoint(model, project, checkpoint, fresh)
    if ckpt:
        console.print(f"Loaded checkpoint: {ckpt}")
    elif fresh:
        console.print("Training from scratch.")

    console.print("Training...")
    meta = run_training(model, project, round_num)

    console.print(f"[green]Round {meta['round']} complete[/green]")
    console.print(
        f"  Train: {meta['num_train']}, Val: {meta['num_val']}, "
        f"Unlabeled: {meta['num_unlabeled']}"
    )
    console.print(f"  Classes: {', '.join(meta['classes'])}")
    if meta["metrics"]:
        for k, v in meta["metrics"].items():
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
    unlabeled_only: bool = typer.Option(True, help="Only push for unlabeled samples"),
    checkpoint: Path | None = typer.Option(None, help="Checkpoint to use (default: latest)"),
    prioritize_uncertain: bool = typer.Option(True, help="Show uncertain images first"),
    refresh: bool = typer.Option(
        False,
        "--refresh/--no-refresh",
        help="Replace existing predictions instead of skipping those tasks",
    ),
    limit: int | None = typer.Option(
        None, help="Push only the top-N predictions (most uncertain first when prioritized)"
    ),
    sample: int | None = typer.Option(
        None, help="Predict on a random subset of N eligible images instead of all of them"
    ),
    use_cache: bool = typer.Option(
        True,
        "--cache/--no-cache",
        help="Use the local task-id cache (--no-cache forces a full refetch, "
        "e.g. after manual prediction changes in LS)",
    ),
) -> None:
    """Push model predictions to Label Studio as pre-annotations."""
    from .active_learning import rank_by_uncertainty
    from .dataset import Sample, load_dataset, split_labeled_unlabeled
    from .predict import run_predictions

    project = _load_project(project_path)
    settings = Settings.load(config_path)
    try:
        project_id = project.require_ls_project_id()
    except ProjectError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    model = project.load_model()
    ckpt = _require_checkpoint(model, project, checkpoint)

    client = _ls_client(settings, project, config_path)
    with console.status("Fetching task list from Label Studio (slow on large projects)..."):
        task_id_map = client.get_task_id_map(project_id, use_cache=use_cache)
        if refresh:
            already_predicted: set[str] = set()
        else:
            unpredicted = client.get_task_id_map(
                project_id, exclude_predicted=True, use_cache=use_cache
            )
            already_predicted = set(task_id_map) - set(unpredicted)

    dataset = load_dataset(project.dataset_path, project.schema)
    samples = (
        split_labeled_unlabeled(dataset)[1]
        if unlabeled_only
        else [s for s in dataset if not s.skipped]
    )
    # Images without an LS task are eligible: tasks are created on demand for
    # whatever makes the final cut. Without --refresh, skip what's already
    # pushed so an interrupted push can resume without re-predicting.
    samples = [s for s in samples if s.path not in already_predicted]
    if not samples:
        console.print("[yellow]Nothing to push — all eligible images have predictions.[/yellow]")
        raise typer.Exit(0)

    if sample is not None and len(samples) > sample:
        # Fresh subset each run so successive rounds see different candidates
        samples = random.sample(samples, sample)
        console.print(f"Sampled {sample} of the eligible images")

    predictions = run_predictions(model, samples, project)
    if prioritize_uncertain:
        predictions = rank_by_uncertainty(predictions)
    if limit is not None:
        predictions = predictions[:limit]

    missing = [p for p in predictions if p.path not in task_id_map]
    if missing:
        console.print(f"Creating {len(missing)} new tasks in Label Studio...")
        client.import_tasks(project_id, [Sample(path=p.path) for p in missing])
        task_id_map = client.get_task_id_map(project_id, use_cache=use_cache)

    num_pushed = _push_with_progress(
        client,
        project_id,
        predictions,
        task_id_map,
        model_version=ckpt.stem,
        replace_existing=refresh,
    )

    skipped = len(predictions) - num_pushed
    console.print(
        f"[green]Pushed {num_pushed} predictions to project {project_id} "
        f"(model version: {ckpt.stem})[/green]"
        + (f" [dim]({skipped} tasks already had predictions)[/dim]" if skipped else "")
    )


@app.command(name="export")
def export_annotations(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
    output: Path | None = typer.Option(None, help="Output path (defaults to the project dataset)"),
) -> None:
    """Export corrected annotations from Label Studio back into the dataset."""
    from .dataset import load_dataset, save_dataset

    project = _load_project(project_path)
    settings = Settings.load(config_path)
    try:
        project_id = project.require_ls_project_id()
    except ProjectError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    client = _ls_client(settings, project, config_path)
    with console.status("Exporting annotations from Label Studio (slow on large projects)..."):
        exported = client.export_annotations(project_id)

    # Merge: LS only holds a subset of the dataset (tasks are created on
    # demand), so overwriting would drop every image without a task
    out_path = output or project.dataset_path
    samples = list(exported)
    if out_path.exists():
        by_path = {e.path: e for e in exported}
        existing = load_dataset(out_path, project.schema)
        samples = [by_path.get(s.path, s) for s in existing]
        known = {s.path for s in existing}
        samples += [e for e in exported if e.path not in known]
    save_dataset(samples, out_path)

    labeled = [s for s in samples if s.is_labeled]
    console.print(
        f"[green]Merged {len(exported)} exported tasks into {out_path} "
        f"({len(samples)} samples, {len(labeled)} labeled)[/green]"
    )

    _warn_undeclared(project, samples)


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
    from strata.catalog import Catalog

    from .to_catalog import MigrationError, describe, migrate

    project = _load_project(project_path)
    if not project.dataset_path.exists():
        console.print(f"[red]No dataset at {project.dataset_path}[/red]")
        raise typer.Exit(1)

    catalog = None if dry_run else Catalog.local(catalog_root)
    try:
        report = migrate(project, catalog, label_set=label_set, dry_run=dry_run)
    except MigrationError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    if dry_run:
        console.print("[yellow]Dry run — nothing was written.[/yellow]")
    console.print(describe(report, catalog_root))
