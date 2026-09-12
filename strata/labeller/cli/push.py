"""The push command: score the pool, order it, and send it with the model's guesses attached."""

from pathlib import Path

import typer

from ..project import ProjectError
from ._remote import _follow, _trainer
from ._shared import (
    ConfigOption,
    ProjectOption,
    _addressing,
    _catalog_config,
    _catalog_for,
    _error,
    _exit_on,
    _label_set_for,
    _load_project,
    _local_paths,
    _ls_client,
    _schema_for,
    _settings,
    _task_map,
    app,
    console,
)


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

    from strata.modelling.remote.client import RemoteError, Trainer

    trainer = Trainer(settings.modelling.url, settings.modelling.token)
    with _exit_on(RemoteError):
        # Asked for by id or not, the host is the one that knows. Checking
        # now costs one request; not checking costs a pool fetched and
        # scored before anything notices.
        found = (
            trainer.run(run_id) if run_id else trainer.latest_run(project.dataset_name, catalog_id)
        )

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
    from strata.modelling.remote.client import RemoteError
    from strata.modelling.remote.service import PredictionRequest

    trainer = _trainer(settings)
    with _exit_on(RemoteError):
        job = trainer.predict(
            PredictionRequest(run_id=run_id, checksums=checksums, features=features or {})
        )

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


@app.command()
def push(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
    limit: int | None = typer.Option(None, help="Review only the top-N, most uncertain first"),
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
    from strata.modelling import PredictionCache, PredictRequest, RunStore
    from strata.modelling import predict as run_predict

    from ..labelstudio.adapter import prediction_to_results
    from ..labelstudio.sync import rebuild_task_map, save_task_map, tasks_to_push
    from ..review import queue
    from ..review.active_learning import STRATEGIES, certainty

    if strategy not in STRATEGIES:
        _error(f"Unknown strategy {strategy!r}. Available: {', '.join(sorted(STRATEGIES))}.")
        raise typer.Exit(1)

    project = _load_project(project_path)
    settings = _settings(config_path)
    catalog, catalog_root = _catalog_for(settings, config_path, name=project.catalog.name)
    label_set_id, _ = _label_set_for(catalog, project)
    addressing = _addressing(settings, _catalog_config(settings, project.catalog.name))
    schema = _schema_for(project, catalog)

    with _exit_on(ProjectError):
        ls_project_id = project.require_ls_project_id(settings.label_studio.url)

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

    pool = catalog.samples.unlabelled(label_set_id, project.collections)
    if not pool:
        console.print("[yellow]Nothing is waiting for review.[/yellow]")
        return

    store = RunStore.local(project.runs_dir)
    remote = bool(settings.modelling.url)
    scores: dict[str, object] = {}
    scoring_run = _run_for_push(settings, project, store, run_id, remote, catalog.id)

    if predictions and scoring_run is not None:
        specs = project.feature_specs
        pool, coverage = queue.feature_values(catalog, pool, specs)
        if coverage.uncovered:
            console.print(
                f"[yellow]{len(coverage.uncovered):,} sample(s) carry no value for "
                f"{', '.join(f.name for f in specs)} and cannot be scored — "
                f"they stay out of this queue until they do.[/yellow]"
            )
        if remote:
            # The host keeps its own cache, keyed on its own run ids — which
            # is the only place that key means anything.
            scores = _remote_predictions(
                settings, scoring_run, [s.checksum for s in pool], coverage.by_checksum
            )
        else:
            with console.status(f"Predicting with run {scoring_run}..."):
                scored = queue.score_locally(
                    store,
                    PredictionCache.local(project.runs_dir),
                    scoring_run,
                    pool,
                    coverage,
                    paths_for=lambda missing: _local_paths(catalog_root, missing),
                    predict=lambda paths, features: run_predict(
                        PredictRequest(run_id=scoring_run, paths=paths, features=features),
                        store,
                    ),
                )
            if scored.reused:
                console.print(
                    f"[dim]{scored.reused:,} prediction(s) reused from run "
                    f"{scoring_run}; {scored.made:,} to make[/dim]"
                )
            scores = scored.scores
    elif predictions:
        console.print("[yellow]No run with a checkpoint yet; pushing without predictions.[/yellow]")

    planned = queue.plan(
        pool,
        scores,
        STRATEGIES[strategy],
        empty_share=empty_share,
        disputed_rows=queue.disputed(
            catalog, label_set_id, project.collections, exclude={s.id for s in pool}
        ),
        limit=limit,
    )
    if planned.disputed:
        console.print(
            f"[yellow]{planned.disputed} sample(s) were answered two ways — "
            f"pushed first so they are looked at again[/yellow]"
        )

    tasks, report = tasks_to_push(
        planned.ranked, catalog, label_set_id, schema, addressing, task_map
    )
    created = client.import_catalog_tasks(ls_project_id, tasks)
    task_map.update(created)
    save_task_map(project, ls_project_id, task_map, catalog.id)

    console.print(
        f"[green]{report.pushed} task(s) created[/green]"
        + (f", {report.already_present} already there" if report.already_present else "")
    )

    if planned.scored:
        payload = [
            (
                s.id,
                prediction_to_results(planned.scored[s.id], schema),
                certainty(planned.scored[s.id]),
            )
            for s in planned.ranked
            if s.id in planned.scored
        ]
        pushed = client.push_catalog_predictions(
            ls_project_id,
            payload,
            task_map,
            model_version=f"run-{scoring_run}",
            replace_existing=refresh,
        )
        console.print(f"[green]{pushed} pre-annotation(s) attached from run {scoring_run}[/green]")
