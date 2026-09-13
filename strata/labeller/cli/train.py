"""The train command: freeze a version, materialise it, train from it, here or on the host."""

from pathlib import Path

import typer

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
    from strata.modelling.remote.client import RemoteError
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
                group_by=project.catalog.group_by or None,
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
