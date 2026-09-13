"""The report command: the training history, or one run in detail."""

import json
from pathlib import Path

import typer
from rich.table import Table

from ._shared import (
    ProjectOption,
    _error,
    _load_project,
    app,
    console,
)


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
