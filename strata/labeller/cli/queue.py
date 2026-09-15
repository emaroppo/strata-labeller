"""Commands over the review queue: fill it, unskip it, export from it, relink it."""

from pathlib import Path

import typer
from rich.markup import escape

from ..project import ProjectError
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

    ls_project_id = project.ls_project_id(settings.label_studio.url)
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
    from ..labelstudio.sync import save_task_map, tasks_to_push

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
        # Only when Label Studio is the one reading files. docs/adr/0013
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
            f"not be mapped. Run 'strata-labeller push --rebuild-map' to recover "
            f"the mapping by listing them.[/yellow]"
        )

    answered = sum(1 for task in tasks if task.answered)
    console.print(
        f"[green]Created Label Studio project '{project.name}' "
        f"(ID: {ls_project_id}) with {len(tasks)} tasks[/green]"
    )
    console.print(f"  {answered} arrived already answered, {len(tasks) - answered} to review")
    console.print(f"  Classes: {', '.join(label_schema.classes)}")


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
    from ..labelstudio.sync import pull_annotations

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
        # Refused before the first write. docs/adr/0030
        _error(
            f"Label(s) nobody declared: {', '.join(sorted(report.undeclared))}. "
            f"Add them with 'strata-labeller class add', then export again."
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
    from ..labelstudio.sync import relink as plan_relink

    project = _load_project(project_path)
    settings = _settings(config_path)
    catalog, _ = _catalog_for(settings, config_path, name=project.catalog.name)
    addressing = _addressing(settings, _catalog_config(settings, project.catalog.name))
    schema = _schema_for(project, catalog)

    client = _ls_client(settings, project, config_path)
    with _exit_on(ProjectError):
        ls_project_id = project.require_ls_project_id(settings.label_studio.url)

    where = addressing.urls.base_url if addressing.urls else f"the {addressing.prefix} mount"
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
    if addressing.urls:
        console.print(
            "[dim]These URLs carry an expiry. Run this again if a queue sits "
            "long enough for images to stop loading.[/dim]"
        )
