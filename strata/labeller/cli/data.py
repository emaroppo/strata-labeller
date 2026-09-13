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
) -> None:
    """Scan the project's data root and register new files in the catalog.

    Which files count, and how they group, comes from the project's sample
    type. Registering is not queueing: the catalog holds the whole pool and
    `push` sends only what you are about to review.
    """
    from strata.catalog import CatalogError

    from .. import corpus

    project = _load_project(project_path)
    settings = _settings(config_path)
    data_dir = project.data_dir
    with _exit_on(ProjectError):
        sample_type = project.sample_type()
    if not data_dir.exists():
        _error(f"Data root does not exist: {data_dir}")
        raise typer.Exit(1)

    catalog, catalog_root = _catalog_for(
        settings, config_path, create=True, name=project.catalog.name
    )
    try:
        label_set_id, _ = catalog.label_sets.get(project.label_set_name)
    except CatalogError:
        label_set_id = catalog.label_sets.create(
            project.label_set_name, project.schema.catalog_schema()
        )
        console.print(f"Created label set '{project.label_set_name}'")

    scanned = corpus.scan(data_dir, sample_type.allows)
    if scanned.skipped:
        console.print(
            f"[yellow]{len(scanned.skipped):,} file(s) skipped — "
            f"{project.sample_type_name} does not admit {', '.join(scanned.kinds)}[/yellow]"
        )
    if not scanned.everything:
        # Nothing there yet is not a mistake: a project exists before its
        # data does, and this is what someone runs to find out.
        console.print(f"[yellow]No files under {data_dir} yet.[/yellow]")
        return
    if not scanned.found:
        # Files, and none of them admitted: the wrong folder, or a type that
        # does not describe what is in it. Returning quietly would report an
        # empty corpus as a success.
        _error(
            f"None of the {len(scanned.everything):,} file(s) under {data_dir} are "
            f"{project.sample_type_name} "
            f"({', '.join('.' + e for e in sorted(sample_type.extensions))}). "
            f"Either the data is elsewhere, or [data] type names the wrong "
            f"thing for it."
        )
        raise typer.Exit(1)

    before, _ = corpus.catalogued(catalog, label_set_id, project.collections)
    with _progress(bar=True, elapsed=True, remaining=True) as progress:
        bar = progress.add_task("Ingesting", total=len(scanned.found))
        try:
            corpus.ingest_files(
                catalog,
                sample_type,
                data_dir,
                scanned.found,
                collections=project.collections,
                batch=batch,
                on_sample=lambda _p: progress.advance(bar),
            )
        except CatalogError as e:
            # A file this type cannot store. The chunks before it are
            # committed, so re-running after fixing it carries on.
            progress.stop()
            _error(str(e))
            raise typer.Exit(1) from None
    after, skipped = corpus.catalogued(catalog, label_set_id, project.collections)
    console.print(
        f"[green]{len(scanned.found)} file(s) scanned, {after - before} new[/green] "
        f"into {catalog_root}"
    )
    console.print(f"  {after + skipped} sample(s) catalogued")


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
    """Convert a corpus into the shape this project's sample type stores.

    Writes the files and an index of what the conversion knew into the
    project's data root. Then run 'ingest', which catalogues them.
    """
    from strata.catalog.types.preparers import PreparerError, available
    from strata.catalog.types.preparers import run as run_preparer

    from .. import corpus

    project = _load_project(project_path)
    source_dir = source or project.source_dir
    if not source_dir.exists():
        _error(
            f"No corpus at {source_dir}. Put the files there, or point "
            f"[data] source_root at where they are."
        )
        raise typer.Exit(1)
    everything = [p for p in sorted(source_dir.rglob("*")) if p.is_file()]
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
                "[dim]Nothing is installed. A conversion is a plugin: "
                "uv pip install strata-prepare-email, or -video.[/dim]"
            )
        raise typer.Exit(1) from None

    preparer = cls()
    scanned = corpus.scan(source_dir, preparer.allows)
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
                preparer, scanned.found, out_dir, on_source=lambda _p: progress.advance(bar)
            )
        except PreparerError as e:
            progress.stop()
            _error(str(e))
            raise typer.Exit(1) from None
    console.print(
        f"[green]{len(scanned.found):,} source file(s) → {len(index.samples):,} "
        f"sample(s)[/green] in {out_dir}"
    )
    # Anything left behind, said out loud.
    for key, count in sorted(preparer.report().items()):
        console.print(f"  {key.replace('_', ' ')}: {count:,}")
    carrying = sum(1 for entry in index.samples.values() if entry.value is not None)
    if carrying:
        console.print(
            f"[dim]{carrying:,} sample(s) came with candidate annotations. They "
            f"are guesses, and nothing lands them in the catalog on its own."
            f"[/dim]"
        )
    console.print(f"\nNext: strata-labeller ingest --project {project.name}")
