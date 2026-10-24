"""The train command: freeze a version, materialise it, train from it, here or on the host."""

from pathlib import Path

import typer

from strata.catalog import CatalogError
from strata.catalog.stages import Context as CatalogContext
from strata.catalog.stages import DatasetRequest, dataset
from strata.modelling.remote.client import RemoteError
from strata.modelling.stages import Context as ModellingContext
from strata.modelling.stages import DatasetIdentity, Host, TrainStageRequest
from strata.modelling.stages import train as train_stage

from ..round import RoundError, describe, run_round
from ._remote import _job_state, _on_another_catalog, _print_run_result, _reattach, _trainer
from ._shared import (
    ConfigOption,
    HostOption,
    ProjectOption,
    _catalog_config,
    _catalog_for,
    _error,
    _exit_on,
    _load_project,
    _modelling,
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
    host_name: str = HostOption,
) -> None:
    """Train on the project's labelled data.

    A round freezes a dataset version in the catalog, materialises it, and
    trains from that directory — so the run resolves back to the exact
    samples behind it, and validation membership is inherited rather than
    recomputed.
    """
    project = _load_project(project_path)

    settings = _settings(config_path)
    catalog, catalog_root = _catalog_for(settings, config_path, name=project.catalog.name)

    host = _modelling(settings, host_name or project.model.host, project.catalog.name)

    if job is not None:
        _reattach(host, job)
        return
    if host.url:
        _remote_round(project, catalog, settings, host, fresh=fresh, val_ratio=val_ratio)
        return

    # Minutes of silence are indistinguishable from a hang. docs/adr/0031
    with (
        _exit_on(RoundError),
        _progress(bar=True, remaining=True, transient=True) as progress,
    ):
        bar = progress.add_task("Materialising", total=None)
        materialising = True

        def tick(done: int, total: int) -> None:
            nonlocal materialising
            progress.update(bar, completed=done, total=total)
            if materialising and done >= total:
                # Training is next: long, quiet and not this bar's business
                materialising = False
                progress.stop()
                console.print(f"Materialised {total:,} sample(s). Training...")

        result = run_round(
            project,
            catalog,
            fresh=fresh,
            val_ratio=val_ratio,
            on_progress=tick,
            # Blobs already on this host, whatever the backend. docs/adr/0002
            cache=catalog_root / "blobs",
        )

    for line in describe(result):
        console.print(line)
    console.print(f"  checkpoint: {result.run.checkpoint}")


def _remote_round(project, catalog, settings, host, fresh: bool, val_ratio: float) -> None:
    """Freeze a dataset here, and have another host train on it. See ``docs/adr/0007``."""
    trainer = _trainer(host)

    # Asked before anything is frozen. docs/adr/0030
    with _exit_on(RemoteError):
        served = trainer.served_catalog()
    if served.get("id") != catalog.id:
        config = _catalog_config(settings, project.catalog.name)
        _error(
            _on_another_catalog(f"The modelling host {host.describe()}", served, catalog, config)
        )
        raise typer.Exit(1)

    with _exit_on(CatalogError):
        frozen = dataset(
            DatasetRequest(
                name=project.dataset_name,
                label_set=project.label_set_name,
                collections=project.collections,
                val_ratio=val_ratio,
                group_by=project.catalog.group_by or None,
                given=project.catalog.given_split,
            ),
            CatalogContext(catalog, project.datasets_dir),
        )
    console.print(f"Dataset {frozen.name} v{frozen.version} → {host.describe()}")

    with _progress(elapsed=True, transient=True) as progress:
        bar = progress.add_task("Waiting for the host...")
        with _exit_on(RemoteError):
            record = train_stage(
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
                    host=Host(host.url, host.token),
                    on_state=_job_state(progress, bar),
                    # The client already spoken to, so a test's fake host is the one asked
                    client=lambda url, token: trainer,
                ),
            )
    _print_run_result(record.run_id, record.parent_run_id, record.materialised, record.metrics)
    console.print(
        "[dim]The run and its checkpoint live on that host, which is where the "
        "next round will warm-start from.[/dim]"
    )
