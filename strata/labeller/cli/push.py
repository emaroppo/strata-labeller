"""The push command: score the pool, order it, and send it with the model's guesses attached."""

from pathlib import Path

import typer

from strata.labels import AnyPrediction, AnyValue
from strata.modelling import RunStore, RunStoreMissing

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


def _store_if_any(project):
    """This project's run store, or None when nothing has trained here.

    Consulted, never made: a push with no runs pushes without predictions,
    and must not leave an empty store behind to say otherwise.
    """
    try:
        return RunStore.open(project.runs_dir)
    except RunStoreMissing:
        return None


def _run_for_push(
    settings, project, store, run_id, remote: bool, catalog_id: str | None = None
) -> str | None:
    """Which run scores this push, in the numbering of whoever will score it.

    Run ids belong to the store that issued them, so the host is asked for
    its own. See ``docs/adr/0005``.
    """
    if not remote:
        if store is None:
            return None
        run = store.get(run_id) if run_id else store.latest(project.dataset_name, catalog_id)
        if run is None or not run.checkpoint:
            return None
        return run.id

    from strata.modelling.remote.client import RemoteError, Trainer

    trainer = Trainer(settings.modelling.url, settings.modelling.token)
    with _exit_on(RemoteError):
        # Asked for by id or not, the host is the one that knows, and it is
        # asked before the pool is fetched. docs/adr/0030
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

    The same job machinery as a round. See ``docs/adr/0007``.
    """
    from pydantic import TypeAdapter

    from strata.modelling.remote.client import RemoteError
    from strata.modelling.remote.wire import PredictionRequest

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
    # emits. docs/adr/0006
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
    review_imports: bool = typer.Option(
        False,
        "--review-imports",
        help=(
            "Spot-review imported labels instead: the samples whose label arrived "
            "with the corpus and nobody confirmed, those the model disagrees with "
            "most first, shown with the imported label as the pre-annotation"
        ),
    ),
) -> None:
    """Send unreviewed samples to Label Studio, least confident first.

    Predictions come from a recorded run, so what a reviewer sees is tied to
    a checkpoint that resolves back to the data behind it.

    With ``--review-imports`` the queue is the imported labels nobody has
    confirmed. Imports are trusted and trained on; this is for tracking
    down the ones a poor result points at. The ranking is how far the
    model's prediction is from the imported label, and what the reviewer
    sees is the import itself, to confirm or correct. Export records which.
    """
    from strata.modelling import PredictionCache, PredictRequest
    from strata.modelling import predict as run_predict

    from ..labelstudio.adapter import prediction_to_results
    from ..labelstudio.sync import rebuild_task_map, save_task_map, tasks_to_push
    from ..review import queue
    from ..review.active_learning import STRATEGIES, against, certainty, disagreement

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

    labels: dict[str, AnyValue] = {}
    if review_imports:
        pool = catalog.samples.unreviewed(label_set_id, project.collections)
        if not pool:
            console.print("[yellow]No imported label is waiting to be confirmed.[/yellow]")
            return
        imported = catalog.annotations.values_of(label_set_id, [s.id for s in pool])
        labels = {s.checksum: imported[s.id] for s in pool if s.id in imported}
    else:
        pool = catalog.samples.unlabelled(label_set_id, project.collections)
        if not pool:
            console.print("[yellow]Nothing is waiting for review.[/yellow]")
            return

    store = _store_if_any(project)
    remote = bool(settings.modelling.url)
    scores: dict[str, AnyPrediction] = {}
    scoring_run = _run_for_push(settings, project, store, run_id, remote, catalog.id)

    if predictions and scoring_run is not None:
        assert store is not None or remote  # a local scoring run came from the store
        specs = project.feature_specs
        pool, coverage = queue.feature_values(catalog, pool, specs)
        if coverage.uncovered:
            console.print(
                f"[yellow]{len(coverage.uncovered):,} sample(s) carry no value for "
                f"{', '.join(f.name for f in specs)} and cannot be scored — "
                f"they stay out of this queue until they do.[/yellow]"
            )
        if remote:
            # The host keeps its own cache, keyed on its own run ids.
            # docs/adr/0006
            scores = _remote_predictions(
                settings, scoring_run, [s.checksum for s in pool], coverage.by_checksum
            )
        else:
            assert store is not None
            with console.status(f"Predicting with run {scoring_run}..."):
                scored = queue.score_locally(
                    store,
                    PredictionCache.beside(store),
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

    if review_imports:
        # Ranked by how far the model is from the imported label, and
        # nothing disputed joins this queue. docs/adr/0028
        ordering = against(labels)
        scores = {c: ordering.bind(c, p) for c, p in scores.items() if c in labels}
        planned = queue.plan(pool, scores, ordering, empty_share=1.0, limit=limit)
        if not scores and limit is not None:
            planned = queue.plan(pool, {}, ordering, empty_share=1.0, limit=limit)
    else:
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

    if review_imports:
        # The reviewer sees the imported label, scored by the model's
        # distance from it. docs/adr/0028
        payload = [
            (
                s.id,
                schema.encode_target(list(labels[s.checksum].values)),
                (
                    disagreement(planned.scored[s.id], labels[s.checksum])
                    if s.id in planned.scored
                    else 1.0
                ),
            )
            for s in planned.ranked
            if s.checksum in labels
        ]
        pushed = client.push_catalog_predictions(
            ls_project_id,
            payload,
            task_map,
            model_version="import" + (f"-vs-run-{scoring_run}" if scoring_run else ""),
            replace_existing=refresh,
        )
        console.print(f"[green]{pushed} imported label(s) attached for review[/green]")
        return

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


@app.command()
def audit(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
    sample: int = typer.Option(..., "--sample", help="How many answered samples to re-queue"),
    seed: int = typer.Option(0, "--seed", help="The draw, so an audit can be repeated"),
    run_id: str | None = typer.Option(
        None, help="Prefer answers that agreed with this run's predictions (default: latest)"
    ),
) -> None:
    """Re-queue answered samples for a blind second look.

    Draws ``--sample`` of the samples a person has answered, takes their
    annotation and any pre-annotation off the task in Label Studio, and
    leaves the task to be answered again from nothing. The first answer
    stays in the catalog's history; export writes the second on top of
    it, and ``report`` counts how many agreed. Where the local prediction
    cache knows what the model showed at push time, the draw is over the
    answers that accepted a pre-annotation unchanged, since those are the
    ones a second look can tell anything about; otherwise it is over
    every answer, and says so.
    """
    import random

    from strata.modelling import PredictionCache

    from ..labelstudio.sync import save_task_map
    from ..review import queue

    project = _load_project(project_path)
    settings = _settings(config_path)
    catalog, _ = _catalog_for(settings, config_path, name=project.catalog.name)
    label_set_id, label_schema = _label_set_for(catalog, project)
    with _exit_on(ProjectError):
        ls_project_id = project.require_ls_project_id(settings.label_studio.url)
    client = _ls_client(settings, project, config_path)
    task_map = _task_map(project, ls_project_id, catalog)

    answered = catalog.samples.labelled(label_set_id, project.collections, source="human")
    answered = [s for s in answered if s.id in task_map]
    if not answered:
        console.print(
            "[yellow]No answered sample has a task in Label Studio to look at again.[/yellow]"
        )
        return
    values = catalog.annotations.values_of(label_set_id, [s.id for s in answered])

    # What the model showed, where this machine recorded it
    store = _store_if_any(project)
    scoring_run = _run_for_push(settings, project, store, run_id, remote=False)
    accepted: list = []
    if scoring_run is not None:
        pool, coverage = queue.feature_values(catalog, answered, project.feature_specs)
        assert store is not None  # a scoring run came from it
        shown = PredictionCache.beside(store).get(
            scoring_run, [s.checksum for s in pool], coverage.digests
        )
        accepted = [
            s
            for s in pool
            if s.checksum in shown
            and label_schema.classes_asserted(shown[s.checksum])
            == label_schema.classes_asserted(values[s.id])
        ]
    if accepted:
        pool, kind = accepted, f"answer(s) that accepted run {scoring_run}'s pre-annotation"
    else:
        pool, kind = answered, "answer(s), since nothing here recorded what the model showed"

    rng = random.Random(seed)
    chosen = rng.sample(pool, min(sample, len(pool)))
    task_ids = [task_map[s.id] for s in chosen]
    client.delete_annotations(ls_project_id, task_ids)
    client.clear_predictions(ls_project_id, task_ids)
    save_task_map(project, ls_project_id, task_map, catalog.id)
    console.print(
        f"[green]{len(chosen)} of {len(pool)} {kind} re-queued blind.[/green] "
        f"Their first answers stay in the catalog; export writes the second on top."
    )
