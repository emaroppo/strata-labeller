"""Commands over the review queue: fill it, push to it, export from it, relink it."""

from pathlib import Path

import typer
from rich.markup import escape

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
    _progress,
    _schema_for,
    _settings,
    _task_map,
    app,
    console,
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

    skipped = catalog.samples.skipped(label_set_id, project.collections)
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
            console.print(f"[dim]{len(selected) - len(task_ids)} had no task there yet.[/dim]")

    moved = catalog.annotations.unskip(label_set_id, [s.id for s in selected])
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
    from ..sync import save_task_map, tasks_to_push

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
    samples = catalog.samples.labelled(
        label_set_id, project.collections
    ) + catalog.samples.unlabelled(label_set_id, project.collections)
    if limit is not None:
        samples = samples[:limit]

    client = _ls_client(settings, project, config_path)
    ls_project_id = client.create_project(project.name)
    if not config.serve_url:
        # Only when Label Studio is the one reading files. Once tasks carry
        # signed URLs to the serving API, a local storage connection points
        # at a mount this deployment no longer has, and configuring one
        # would suggest the mount still matters.
        client.setup_local_storage(ls_project_id, path=f"/label-studio/data/{config.blobs_prefix}")

    tasks, _ = tasks_to_push(
        samples, catalog, label_set_id, schema, _addressing(settings, config), {}
    )
    with _progress(bar=True, elapsed=True) as progress:
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

    from strata.modelling.client import RemoteError, Trainer

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
    from strata.modelling.client import RemoteError
    from strata.modelling.service import PredictionRequest

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

    from .. import queue
    from ..active_learning import STRATEGIES, certainty
    from ..adapter import prediction_to_results
    from ..sync import rebuild_task_map, save_task_map, tasks_to_push

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
    from ..sync import pull_annotations

    project = _load_project(project_path)
    settings = _settings(config_path)
    catalog, _ = _catalog_for(settings, config_path, name=project.catalog.name)
    label_set_id, label_schema = _label_set_for(catalog, project)
    schema = _schema_for(project, catalog)

    with _exit_on(ProjectError):
        ls_project_id = project.require_ls_project_id(settings.label_studio.url)

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

    written = catalog.annotations.annotate_many(label_set_id, items, source="human")
    console.print(
        f"[green]{written.annotated} annotation(s) and {written.skipped} skip(s) "
        f"into the catalog[/green]"
    )
    if report.unrecognised:
        console.print(
            f"[yellow]{len(report.unrecognised)} task(s) point at nothing this "
            f"catalog knows, and were left alone.[/yellow]"
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
    from ..sync import relink as plan_relink

    project = _load_project(project_path)
    settings = _settings(config_path)
    catalog, _ = _catalog_for(settings, config_path, name=project.catalog.name)
    addressing = _addressing(settings, _catalog_config(settings, project.catalog.name))
    schema = _schema_for(project, catalog)

    client = _ls_client(settings, project, config_path)
    with _exit_on(ProjectError):
        ls_project_id = project.require_ls_project_id(settings.label_studio.url)

    where = addressing.base_url or f"the {addressing.prefix} mount"
    console.print(f"[bold]Label Studio project {ls_project_id}[/bold] → {where}\n")

    with console.status("Listing tasks..."):
        tasks = client.list_tasks(ls_project_id)
    report = plan_relink(tasks, catalog, addressing, schema.data_key)

    console.print(
        f"  {report.total:,} task(s): {len(report.changes):,} to repoint, "
        f"{report.unchanged:,} already current"
    )
    if report.unrecognised:
        console.print(
            f"  [yellow]{len(report.unrecognised):,} name no sample and are left alone[/yellow]"
        )
        for url in report.unrecognised[:3]:
            console.print(f"    [dim]{escape(url)}[/dim]")

    if dry_run:
        console.print("\n[dim]Nothing was changed.[/dim]")
        return
    if not report.changes:
        console.print("\n[green]Nothing to do.[/green]")
        return

    with _progress(bar=True, remaining=True) as progress:
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
