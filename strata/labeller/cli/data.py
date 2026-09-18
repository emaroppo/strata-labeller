"""Commands that get a corpus in: prepare, then ingest."""

from pathlib import Path

import typer

from ..project import ProjectError
from ._shared import (
    ConfigOption,
    ProjectOption,
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
def ingest(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
    batch: int = typer.Option(2000, help="Files per transaction"),
    import_name: str | None = typer.Option(
        None,
        "--import",
        help=(
            "Land the labels the prepared corpus carries, under this batch name. "
            "Required when the index beside the files labels any of them"
        ),
    ),
) -> None:
    """Register the prepared corpus in the project's data root in the catalog.

    What is registered is what the prepared index names, each file checked
    against the project's sample type first: a corpus with any file short
    of it is refused whole, before anything is written. Labels the corpus
    arrived with land in the same pass, as an import batch: trusted,
    trained on, and spot-reviewed with ``push --review-imports``.
    Registering is not queueing: the catalog holds the whole pool and
    ``push`` sends only what you are about to review.
    """
    from strata.catalog import CatalogError, IntakeError, admit
    from strata.contracts import PREPARED_NAME

    from .. import corpus

    project = _load_project(project_path)
    settings = _settings(config_path)
    data_dir = project.data_dir
    with _exit_on(ProjectError):
        sample_type = project.sample_type()
    if not data_dir.exists():
        _error(f"Data root does not exist: {data_dir}")
        raise typer.Exit(1)
    if not (data_dir / PREPARED_NAME).exists():
        if not any(p.is_file() for p in data_dir.rglob("*")):
            # Nothing there yet is not a mistake: a project exists before its
            # data does, and this is what someone runs to find out.
            console.print(f"[yellow]No files under {data_dir} yet.[/yellow]")
            return
        # Files, and nothing saying what they are. docs/adr/0040
        _error(
            f"{data_dir} has files and no {PREPARED_NAME} saying what they are. "
            f"Prepare them first: {_prepare_hint(project)}"
        )
        raise typer.Exit(1)

    try:
        admission = admit(data_dir, sample_type, project.sample_type_name)
    except IntakeError as e:
        _error(str(e))
        raise typer.Exit(1) from None
    if admission.unindexed:
        console.print(
            f"[yellow]{admission.unindexed:,} file(s) under {data_dir} are not in "
            f"the prepared index, and were left out[/yellow]"
        )
    if not admission.entries:
        console.print(f"[yellow]The prepared index in {data_dir} names no files.[/yellow]")
        return
    if admission.values and import_name is None:
        # Said before anything is written. docs/adr/0028
        _error(
            f"The prepared corpus labels {len(admission.values):,} file(s). Name the import "
            f"to land them with the files: --import <batch>."
        )
        raise typer.Exit(1)

    catalog, catalog_root = _catalog_for(
        settings, config_path, create=True, name=project.catalog.name
    )
    try:
        label_set_id, _ = catalog.label_sets.get(project.label_set_name)
    except CatalogError:
        label_set_id = catalog.label_sets.create(project.label_set_name, project.label_set.schema)
        console.print(f"Created label set '{project.label_set_name}'")

    before, _ = corpus.catalogued(catalog, label_set_id, project.collections)
    with _progress(bar=True, elapsed=True, remaining=True) as progress:
        bar = progress.add_task("Ingesting", total=len(admission.entries))
        try:
            registered = corpus.ingest_files(
                catalog,
                sample_type,
                admission,
                collections=project.collections,
                batch=batch,
                on_sample=lambda _p: progress.advance(bar),
            )
        except CatalogError as e:
            # A file this type cannot store; the chunks before it are
            # committed. docs/adr/0032
            progress.stop()
            _error(str(e))
            raise typer.Exit(1) from None
    after, skipped = corpus.catalogued(catalog, label_set_id, project.collections)
    console.print(
        f"[green]{len(admission.entries):,} file(s) prepared, {after - before} new[/green] "
        f"into {catalog_root}"
    )
    console.print(f"  {after + skipped} sample(s) catalogued")

    if admission.values and import_name is not None:
        written = corpus.land_labels(catalog, label_set_id, admission, registered, import_name)
        if written is not None:
            console.print(
                f"[green]{written.annotated} label(s) landed as import {import_name!r}[/green]"
                + (f", {written.kept} left alone — a person had answered" if written.kept else "")
            )


def _prepare_hint(project) -> str:
    """The command that would prepare this project's corpus, as near as can be said."""
    folder = {"image": "image-folder", "frames": "frames-folder", "text": "text-folder"}
    name = folder.get(project.sample_type_name)
    if name is None:
        return f"strata-labeller prepare --project {project.name}"
    return (
        f"strata-labeller prepare --project {project.name} --preparer {name} "
        f"--from {project.data_dir}  (files already in place are indexed, not copied)"
    )


@app.command()
def prepare(
    project_path: Path | None = ProjectOption,
    source: Path | None = typer.Option(
        None, "--from", help="Where the corpus is; defaults to the project's source_root"
    ),
    preparer_name: str = typer.Option(
        "", "--preparer", help="Which conversion to run; defaults to the project's setting"
    ),
) -> None:
    """Get a corpus into the shape this project's sample type takes.

    Writes the files and a prepared index of what the preparer knew into
    the project's data root. Then run 'ingest', which checks the index
    against the type and catalogues what it names. Files already in the
    data root are indexed where they are.
    """
    from strata.contracts import PREPARED_NAME
    from strata.prepare import PreparerError, available
    from strata.prepare import run as run_preparer

    from .. import corpus

    project = _load_project(project_path)
    source_dir = source or project.source_dir
    if not source_dir.exists():
        _error(
            f"No corpus at {source_dir}. Put the files there, or point "
            f"[data] source_root at where they are."
        )
        raise typer.Exit(1)
    # The index from an earlier run is not a source, where the files are
    # prepared in place. docs/adr/0040
    everything = [
        p for p in sorted(source_dir.rglob("*")) if p.is_file() and p != source_dir / PREPARED_NAME
    ]
    if not everything:
        console.print(f"[yellow]No files under {source_dir} yet.[/yellow]")
        return

    try:
        cls = corpus.choose_preparer(
            preparer_name or project.data.preparer, project.sample_type_name, everything[0]
        )
    except (PreparerError, ProjectError) as e:
        _error(str(e))
        if not available():
            console.print(
                "[dim]Nothing is installed. strata-prepare carries the folder "
                "preparers, and a conversion is a plugin: strata-prepare-video.[/dim]"
            )
        raise typer.Exit(1) from None

    preparer = cls()
    scanned = corpus.scan(source_dir, preparer.allows, ignore=[source_dir / PREPARED_NAME])
    if scanned.skipped:
        console.print(
            f"[yellow]{len(scanned.skipped):,} file(s) skipped — '{cls.name}' does "
            f"not read {', '.join(scanned.kinds)}[/yellow]"
        )
    if not scanned.found:
        _error(
            f"None of the {len(everything):,} file(s) under {source_dir} are "
            f"read by '{cls.name}' "
            f"({', '.join('.' + e for e in sorted(cls.sources))})."
        )
        raise typer.Exit(1)

    out_dir = project.data_dir
    with _progress(bar=True, elapsed=True, remaining=True) as progress:
        bar = progress.add_task(f"Preparing with '{cls.name}'", total=len(scanned.found))
        try:
            index = run_preparer(
                preparer,
                scanned.found,
                out_dir,
                root=source_dir,
                on_source=lambda _p: progress.advance(bar),
            )
        except PreparerError as e:
            progress.stop()
            _error(str(e))
            raise typer.Exit(1) from None
    console.print(
        f"[green]{len(scanned.found):,} source file(s) → {len(index.samples):,} "
        f"sample(s)[/green] in {out_dir}"
    )
    # Anything left behind, said out loud. docs/adr/0036
    for key, count in sorted(preparer.report().items()):
        console.print(f"  {key.replace('_', ' ')}: {count:,}")
    carrying = sum(1 for entry in index.samples.values() if entry.value is not None)
    if carrying:
        console.print(
            f"[dim]{carrying:,} sample(s) came with candidate annotations. They land "
            f"as an import, trusted until a person looks: ingest --import <batch>."
            f"[/dim]"
        )
    console.print(f"\nNext: strata-labeller ingest --project {project.name}")
