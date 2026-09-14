"""Commands over a project: make one, list them, extend its classes."""

from pathlib import Path

import typer
from rich.markup import escape
from rich.table import Table

from strata.catalog import Catalog

from ..project import PROJECTS_DIR, LabellingProject, ProjectError
from ._shared import (
    ConfigOption,
    ProjectOption,
    _catalog_for,
    _catalog_if_any,
    _error,
    _exit_on,
    _label_set_for,
    _load_project,
    _ls_client,
    _settings,
    app,
    console,
)


@app.command()
def new(
    name_or_path: str = typer.Argument(
        ..., metavar="NAME", help=f"Project name (created under {PROJECTS_DIR}/) or a path"
    ),
    name: str | None = typer.Option(None, help="Project name (default: directory name)"),
    classes: list[str] = typer.Option([], "--class", help="Label class (repeatable)"),
    single: bool = typer.Option(False, "--single", help="Classes are mutually exclusive"),
    task: str = typer.Option(
        "classification", help="What is annotated: classification, bbox or span"
    ),
    sample_type: str = typer.Option(
        "image", "--type", help="A registered sample type (see 'strata-catalog types')"
    ),
    custom: bool = typer.Option(
        False, "--custom", help="Start a labeling config of the project's own"
    ),
) -> None:
    """Scaffold a new project under projects/ (or at an explicit path)."""
    directory = Path(name_or_path)
    # A bare name lands in projects/; anything path-shaped is taken literally
    if len(directory.parts) == 1 and not directory.is_absolute():
        directory = Path(PROJECTS_DIR) / directory
    directory.mkdir(parents=True, exist_ok=True)
    with _exit_on(ProjectError):
        project = LabellingProject.create(
            directory,
            name=name,
            classes=list(classes),
            choice="single" if single else "multiple",
            task=task,
            sample_type=sample_type,
            custom=custom,
        )

    console.print(f"[green]Created project '{project.name}' in {directory}[/green]")
    console.print(f"  Put {project.schema.media.name} files in {project.data_dir}, then run:")
    console.print(f"    strata-labeller ingest --project {project.name}")


@app.command()
def templates() -> None:
    """List the labeling templates: which task over which media."""
    from ..labelstudio import schemas

    table = Table(title="Labeling templates")
    table.add_column("Template", style="cyan")
    table.add_column("Task", style="magenta")
    table.add_column("Media", style="magenta")
    table.add_column("Annotations", style="green")
    for name, spec in sorted(schemas.TEMPLATES.items()):
        table.add_row(name, spec.schema.task, spec.media.name, spec.control_tag)
    console.print(table)
    console.print(
        "[dim]A project picks one by its [label_set] task and the media of its "
        "[data] type; a [label_studio] config of its own supplies the layout.[/dim]"
    )


@app.command(name="projects")
def list_projects_cmd() -> None:
    """List the projects under projects/."""
    from strata.catalog import CatalogError

    from ..project import list_projects

    found = list_projects()
    if not found:
        console.print(
            "[yellow]No projects yet. Create one with 'strata-labeller new <name>'.[/yellow]"
        )
        return

    # Counts live in the catalog now, and a project can be listed without
    # one — a project that has never ingested is still a project.
    #
    # Opened per project rather than once: two projects on one host need
    # not draw from the same catalog, and counting both against whichever
    # happens to be the default is how a listing reports numbers that
    # belong to another corpus.
    settings = _settings()
    catalogs: dict[str, Catalog | None] = {}

    def catalog_for_project(project):
        name = project.catalog.name
        if name not in catalogs:
            catalogs[name] = _catalog_if_any(settings, name)
        return catalogs[name]

    table = Table(title="Projects")
    table.add_column("Name", style="cyan")
    table.add_column("Schema", style="magenta")
    table.add_column("Classes", style="green")
    table.add_column("Samples", justify="right")
    table.add_column("Labeled", justify="right")
    table.add_column("LS id", justify="right")

    ls_url = settings.label_studio.url
    for directory in found:
        try:
            project = LabellingProject.load(directory)
        except ProjectError as e:
            table.add_row(directory.name, "-", f"[red]{escape(str(e))}[/red]", "-", "-", "-")
            continue
        total, labeled = "-", "-"
        catalog = catalog_for_project(project)
        if catalog is not None:
            try:
                label_set_id, _ = catalog.label_sets.get(project.label_set_name)
            except CatalogError:
                pass
            else:
                where = project.collections
                annotated = catalog.samples.labelled(label_set_id, where)
                queued = catalog.samples.unlabelled(label_set_id, where)
                skipped = catalog.samples.skipped(label_set_id, where)
                total = str(len(annotated) + len(queued) + len(skipped))
                labeled = str(len(annotated))
        table.add_row(
            project.name,
            project.schema.type,
            ", ".join(project.schema.classes) or "-",
            total,
            labeled,
            str(project.ls_project_id(ls_url) or "-"),
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
    from ..labelstudio.label_config import LabelConfigError

    project = _load_project(project_path)
    settings = _settings(config_path)

    with _exit_on(ProjectError):
        # No inference from what is in use: the label set holds the class
        # list now, and guessing it from annotations was how an unpinned
        # project got one before there was anywhere to pin it
        classes = project.add_classes(names)
    console.print(f"[green]Added {', '.join(names)}[/green] — classes: {', '.join(classes)}")

    _add_to_label_set(project, settings, classes)

    ls_project_id = project.ls_project_id(settings.label_studio.url)
    if push and ls_project_id is not None:
        client = _ls_client(settings, project, config_path)
        for name in names:
            try:
                client.add_class_to_config(ls_project_id, name)
            except LabelConfigError as e:
                _error(f"Label Studio config not updated: {e}")
                console.print(
                    "[yellow]project.toml is updated; add the class in the Label "
                    "Studio UI to match.[/yellow]"
                )
                raise typer.Exit(1) from None
        console.print(f"Label Studio project {ls_project_id} updated — refresh the tab to see it.")

    from strata.catalog import CatalogError

    catalog = _catalog_if_any(settings, project.catalog.name)
    if catalog is not None:
        try:
            label_set_id, _ = catalog.label_sets.get(project.label_set_name)
        except CatalogError:
            return
        labelled = len(catalog.samples.labelled(label_set_id, project.collections))
        skipped = len(catalog.samples.skipped(label_set_id, project.collections))
        if labelled:
            console.print(
                f"[dim]{labelled} sample(s) were labelled before this class existed.[/dim]"
            )
        if skipped:
            console.print(
                f"[dim]{skipped} skipped sample(s) may contain it — "
                f"'strata-labeller unskip' returns them to the queue.[/dim]"
            )


def _add_to_label_set(project: LabellingProject, settings, classes: list[str]) -> None:
    """Widen the catalog's label set to match the project's class list.

    Both have to move together: the labeling config Label Studio renders
    comes from project.toml, while what an export is validated against comes
    from the label set. A class in one and not the other means a reviewer can
    apply a label the catalog will then refuse.
    """
    from strata.catalog import CatalogError

    catalog = _catalog_if_any(settings, project.catalog.name)
    if catalog is None:
        return
    try:
        label_set_id, schema = catalog.label_sets.get(project.label_set_name)
    except CatalogError:
        # No label set yet: ingest creates it from project.toml, so the
        # classes arrive with it
        return

    added = [c for c in classes if c not in schema.classes]
    if not added:
        return
    # Append-only, because a checkpoint maps output neurons to this list by
    # position and a run records the list it trained with
    catalog.label_sets.set_classes(
        label_set_id, schema.model_copy(update={"classes": list(classes)})
    )
    console.print(f"Label set '{project.label_set_name}' widened by {', '.join(added)}")


@class_app.command("list")
def class_list(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
) -> None:
    """List the label set's classes with how many samples carry each."""
    project = _load_project(project_path)
    settings = _settings(config_path)
    catalog, _ = _catalog_for(settings, config_path, name=project.catalog.name)
    label_set_id, schema = _label_set_for(catalog, project)

    table = Table(title=f"Classes — {project.label_set_name}")
    table.add_column("Class", style="cyan")
    table.add_column("Samples", justify="right", style="green")
    for name in schema.classes:
        # From the class index rather than by decoding every annotation,
        # which is what that table exists for
        table.add_row(
            name, str(len(catalog.samples.with_class(label_set_id, name, project.collections)))
        )
    console.print(table)

    declared = set(project.label_set.classes)
    drifted = set(schema.classes) ^ declared
    if declared and drifted:
        # Label Studio renders its config from project.toml while an export
        # is validated against the label set, so a class in one and not the
        # other lets a reviewer apply a label the catalog then refuses
        console.print(
            f"[yellow]project.toml and the label set disagree: "
            f"{', '.join(sorted(drifted))}[/yellow]"
        )
