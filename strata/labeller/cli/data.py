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

    Which files count comes from the project's sample type, so a text
    project picks up documents where an image project picks up pictures.
    Grouping comes from the same place: frames of one video move together,
    so near-duplicates cannot straddle a train/val split. A corpus written
    by 'prepare' says which group each sample is in, and the type reads it.

    Registering is not queueing. The catalog holds the whole pool and `push`
    sends only what you are about to review, so there is no cost to
    cataloguing everything.
    """
    from strata.catalog import CatalogError

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

    # Everything under the root, then checked. Filtering on the way in is how
    # a corpus ends up quietly smaller than the directory it came from.
    everything = [p for p in sorted(data_dir.rglob("*")) if p.is_file()]
    found = [p for p in everything if sample_type.allows(p)]
    skipped = [p for p in everything if p not in set(found)]

    if skipped:
        kinds = sorted({p.suffix.lower() or "(none)" for p in skipped})
        console.print(
            f"[yellow]{len(skipped):,} file(s) skipped — "
            f"{project.sample_type_name} does not admit {', '.join(kinds)}[/yellow]"
        )
    if not everything:
        # Nothing there yet is not a mistake: a project exists before its
        # data does, and this is what someone runs to find out.
        console.print(f"[yellow]No files under {data_dir} yet.[/yellow]")
        return
    if not found:
        # Files, and none of them admitted. Something is wrong — the wrong
        # folder, or a type that does not describe what is in it — and
        # returning quietly would report an empty corpus as a success.
        _error(
            f"None of the {len(everything):,} file(s) under {data_dir} are "
            f"{project.sample_type_name} "
            f"({', '.join('.' + e for e in sorted(sample_type.extensions))}). "
            f"Either the data is elsewhere, or [data] type names the wrong "
            f"thing for it."
        )
        raise typer.Exit(1)

    where = project.collections
    before = len(catalog.samples.unlabelled(label_set_id, where)) + len(
        catalog.samples.labelled(label_set_id, where)
    )
    # Grouped, because a group is one transaction and one group_id. Ungrouped
    # files share a bucket, so a plain image project is a handful of batches
    # rather than one per file.
    by_group: dict[str | None, list[Path]] = {}
    for path in found:
        by_group.setdefault(sample_type.group_id_for(path, data_dir), []).append(path)

    with _progress(bar=True, elapsed=True, remaining=True) as progress:
        bar = progress.add_task("Ingesting", total=len(found))
        for group_id, paths in by_group.items():
            for i in range(0, len(paths), batch):
                chunk = paths[i : i + batch]
                sources = {p: str(p.relative_to(data_dir)) for p in chunk}
                try:
                    catalog.ingest(
                        chunk,
                        media=sample_type.media,
                        subtype=type(sample_type).subtype(),
                        group_id=group_id,
                        # What only the type knows, plus where it came from
                        metadata_for=lambda p: {
                            "source_path": sources[p],
                            **sample_type.metadata_for(p, data_dir),
                        },
                        # Only for a type that has a canonical form; for the
                        # rest this is None and no file is read twice.
                        canonicalise=(
                            sample_type.canonicalise
                            if type(sample_type).canonicalises()
                            else None
                        ),
                        collections=project.collections,
                        on_sample=lambda _p: progress.advance(bar),
                    )
                except CatalogError as e:
                    # A file this type cannot store — a document in an
                    # encoding it will not guess at. The chunks before this
                    # one are committed, so re-running after fixing it
                    # carries on rather than starting over.
                    progress.stop()
                    _error(str(e))
                    raise typer.Exit(1) from None

    after = len(catalog.samples.unlabelled(label_set_id, where)) + len(
        catalog.samples.labelled(label_set_id, where)
    )
    skipped_count = len(catalog.samples.skipped(label_set_id, where))
    console.print(
        f"[green]{len(found)} file(s) scanned, {after - before} new[/green] "
        f"into {catalog_root}"
    )
    console.print(
        f"  {after + skipped_count} sample(s) catalogued, "
        f"{len(by_group)} group(s) touched"
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
    """Convert a corpus into the shape this project's sample type stores.

    Writes the files and an index of what the conversion knew into the
    project's data root. Then run 'ingest', which catalogues them.
    """
    from strata.catalog.types.preparers import PreparerError, available, for_source, resolve
    from strata.catalog.types.preparers import run as run_preparer

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

    name = preparer_name or project.data.preparer
    try:
        wants = project.sample_type_name
        cls = resolve(name) if name else for_source(wants, everything[0])
    except (PreparerError, ProjectError) as e:
        _error(str(e))
        if not available():
            console.print(
                "[dim]Nothing is installed. A conversion is a plugin: "
                "uv pip install strata-prepare-email, or -video.[/dim]"
            )
        raise typer.Exit(1) from None

    if cls.produces != project.sample_type_name:
        # The corpus would convert, and ingest would then admit none of it
        _error(
            f"'{cls.name}' produces '{cls.produces}' samples and this project "
            f"ingests '{project.sample_type_name}'."
        )
        raise typer.Exit(1)

    preparer = cls()
    found = [p for p in everything if preparer.allows(p)]
    skipped = [p for p in everything if p not in set(found)]
    if skipped:
        kinds = sorted({p.suffix.lower() or "(none)" for p in skipped})
        console.print(
            f"[yellow]{len(skipped):,} file(s) skipped — '{cls.name}' does "
            f"not read {', '.join(kinds)}[/yellow]"
        )
    if not found:
        _error(
            f"None of the {len(everything):,} file(s) under {source_dir} are "
            f"read by '{cls.name}' "
            f"({', '.join('.' + e for e in sorted(cls.sources))})."
        )
        raise typer.Exit(1)

    out_dir = project.data_dir
    with _progress(bar=True, elapsed=True, remaining=True) as progress:
        bar = progress.add_task(f"Preparing with '{cls.name}'", total=len(found))
        try:
            index = run_preparer(
                preparer, found, out_dir, on_source=lambda _p: progress.advance(bar)
            )
        except PreparerError as e:
            progress.stop()
            _error(str(e))
            raise typer.Exit(1) from None

    console.print(
        f"[green]{len(found):,} source file(s) → {len(index.samples):,} "
        f"sample(s)[/green] in {out_dir}"
    )
    # Anything left behind, said out loud. A corpus that arrives quietly
    # smaller than the source it came from is the failure ingest already
    # goes out of its way to avoid.
    for key, count in sorted(preparer.report().items()):
        console.print(f"  {key.replace('_', ' ')}: {count:,}")
    carrying = sum(1 for entry in index.samples.values() if entry.value is not None)
    if carrying:
        console.print(
            f"[dim]{carrying:,} sample(s) came with candidate annotations. They "
            f"are guesses, and nothing lands them in the catalog on its own."
            f"[/dim]"
        )
    console.print(f"\nNext: auto-labeller ingest --project {project.name}")
