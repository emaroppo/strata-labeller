"""Commands that train and read the history: train, report, import-rounds."""

import json
from pathlib import Path

import typer
from rich.table import Table

from ._remote import _job_state, _on_another_catalog, _reattach, _trainer
from ._shared import (
    ConfigOption,
    ProjectOption,
    _catalog_config,
    _catalog_for,
    _error,
    _exit_on,
    _load_project,
    _progress,
    _settings,
    app,
    console,
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


    from ..round import RoundError, describe, run_round

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
        with _progress(bar=True, remaining=True, transient=True) as progress:
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


def _remote_round(project, catalog, settings, fresh: bool, val_ratio: float) -> None:
    """Freeze a dataset here, and have another host train on it.

    The split is where the knowledge is. Which samples make a dataset is the
    project's business — its collections, its label set, its val ratio — and
    the catalog is reachable from both machines. Everything after that needs
    a GPU and the checkpoints, and both live there.
    """
    from strata.catalog import CatalogError
    from strata.catalog.stages import Context as CatalogContext
    from strata.catalog.stages import DatasetRequest, dataset
    from strata.modelling.client import RemoteError
    from strata.modelling.stages import Context as ModellingContext
    from strata.modelling.stages import DatasetIdentity, Host, TrainStageRequest, train

    trainer = _trainer(settings)

    # Asked before anything is frozen. A host on another catalog would refuse
    # the round anyway; asking first says which machine to repoint, and
    # leaves no dataset version behind for a round that never ran.
    with _exit_on(RemoteError):
        served = trainer.served_catalog()
    if served.get("id") != catalog.id:
        config = _catalog_config(settings, project.catalog.name)
        _error(_on_another_catalog("The modelling host", served, catalog, config))
        raise typer.Exit(1)

    with _exit_on(CatalogError):
        frozen = dataset(
            DatasetRequest(
                name=project.dataset_name,
                label_set=project.label_set_name,
                collections=project.collections,
                val_ratio=val_ratio,
            ),
            CatalogContext(catalog, project.datasets_dir),
        )
    console.print(f"Dataset {frozen.name} v{frozen.version} → {settings.modelling.url}")

    with _progress(elapsed=True, transient=True) as progress:
        bar = progress.add_task("Waiting for the host...")
        with _exit_on(RemoteError):
            record = train(
                TrainStageRequest(
                    dataset=DatasetIdentity(
                        dataset_id=frozen.dataset_id,
                        name=frozen.name,
                        version=frozen.version,
                        annotation_digest=frozen.annotation_digest,
                        catalog_id=frozen.catalog_id,
                    ),
                    model=project.model.ref,
                    params=project.model.params,
                    fresh_params=project.model.fresh_params,
                    fresh=fresh,
                    features=[spec.as_dict() for spec in project.feature_specs],
                ),
                ModellingContext(
                    store=None,
                    host=Host(settings.modelling.url, settings.modelling.token),
                    on_state=_job_state(progress, bar),
                    # The client already spoken to, so the handshake is not
                    # repeated and a test's fake host is the one asked
                    client=lambda url, token: trainer,
                ),
            )
    _print_train_record(record)
    console.print(
        "[dim]The run and its checkpoint live on that host, which is where the "
        "next round will warm-start from.[/dim]"
    )


def _print_train_record(record) -> None:
    """A finished remote round: the run, its parent, and its metrics."""
    console.print(
        f"[green]Run {record.run_id}[/green]"
        + (f", continuing run {record.parent_run_id}" if record.parent_run_id else " (cold)")
    )
    if record.materialised:
        console.print(f"  [dim]{record.materialised:,} sample(s) materialised there[/dim]")
    for metric, value in sorted(record.metrics.items()):
        console.print(f"  {metric}: {value}")


#: What "how is it going" means per task. A model may report anything it
#: likes alongside; this is only which one the history plots by default.
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
    across rounds is one query. ``--json`` writes the same query to stdout
    and nothing else, so it pipes; it carries every metric and each run's
    ``params`` and ``classes``, which are what let a consumer decide
    whether two runs are even comparable. A change is shown only where one
    run actually continues the one above.
    """
    from strata.modelling import RunStore

    from .. import history

    project = _load_project(project_path)
    if not (project.runs_dir / "runs.db").exists():
        _error(f"No runs recorded at {project.runs_dir}. Run 'train' first.")
        raise typer.Exit(1)

    store = RunStore.local(project.runs_dir)
    metric = metric or history.headline_metric(project.schema.task)

    if run_id is not None:
        run = store.get(run_id)
        if run is None:
            _error(f"No run with id {run_id}")
            raise typer.Exit(1)
        if as_json:
            _emit_json(history.run_detail(store, run))
            return
        _print_run(store, run)
        return

    rows = history.history(store, project.dataset_name, metric)
    if not rows:
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

    if as_json:
        _emit_json(history.history_json(project.dataset_name, metric, rows))
        return

    table = Table(title=f"{project.dataset_name} — {metric}")
    table.add_column("Run", justify="right")
    table.add_column("Dataset", justify="right")
    table.add_column(metric, justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("Lineage")
    for row in rows:
        # Whether it continued, not what from: two ids in one row of a table
        # is a row of ellipses, and the chain itself is what `--run` is for.
        lineage = "warm" if row.warm else "[yellow]unchained[/yellow]"
        shown = f"v{row.version}" if row.version is not None else "[dim]—[/dim]"
        delta = f"{row.delta:+.4f}" if row.delta is not None else ""
        table.add_row(row.run.short, shown, f"{row.value:.4f}", delta, lineage)
    console.print(table)


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
    from ..import_rounds import describe, import_rounds, read_rounds

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
