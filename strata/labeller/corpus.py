"""Getting a project's files into its catalog: what a directory holds, and registering it.

What ``prepare`` and ``ingest`` do between reading a directory and saying
what happened. Nothing here prints; each step returns what it found and
what it left out, so a corpus never ends up quietly smaller than the
directory it came from.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Scan:
    """Every file under a root, split by whether something admits it."""

    everything: list[Path]
    found: list[Path]
    skipped: list[Path]

    @property
    def kinds(self) -> list[str]:
        """The extensions that were skipped, for saying which."""
        return sorted({p.suffix.lower() or "(none)" for p in self.skipped})


def scan(root: Path, allows: Callable[[Path], bool]) -> Scan:
    """Everything under ``root``, then checked.

    Filtering on the way in is how a corpus ends up quietly smaller than
    the directory it came from.
    """
    everything = [p for p in sorted(Path(root).rglob("*")) if p.is_file()]
    found = [p for p in everything if allows(p)]
    admitted = set(found)
    return Scan(everything, found, [p for p in everything if p not in admitted])


def ingest_files(
    catalog,
    sample_type,
    data_dir: Path,
    found: Iterable[Path],
    *,
    collections,
    batch: int,
    on_sample: Callable[[Path], None] | None = None,
) -> dict[Path, int]:
    """Register ``found`` in ``catalog`` as ``sample_type`` says; returns each file's sample id.

    In batches, because a batch is one transaction: a chunk that fails
    leaves the ones before it committed, so a re-run after fixing the file
    carries on. What a sample is grouped by, if anything, is in the metadata
    the type records — a frame's video — and nothing here treats it apart.
    """
    paths = list(found)
    canonicalise = sample_type.canonicalise if type(sample_type).canonicalises() else None
    registered: dict[Path, int] = {}
    for start in range(0, len(paths), batch):
        chunk = paths[start : start + batch]
        sources = {p: str(p.relative_to(data_dir)) for p in chunk}
        ids = catalog.ingest(
            chunk,
            media=sample_type.media,
            subtype=type(sample_type).subtype(),
            # What only the type knows, plus where it came from
            metadata_for=lambda p: {
                "source_path": sources[p],
                **sample_type.metadata_for(p, data_dir),
            },
            canonicalise=canonicalise,
            collections=collections,
            on_sample=on_sample,
        )
        registered.update(zip(chunk, ids, strict=True))
    return registered


def land_labels(catalog, label_set_id: int, data_dir: Path, registered: dict[Path, int], batch):
    """Store the labels the prepared index carries for ``registered`` files, as one import.

    A corpus that arrives labelled is labelled the moment it is catalogued:
    the index beside the files is what a preparer wrote about them, and its
    labels enter with them, under ``source="import"`` and the batch name
    given. A file the index labels but ingest did not register is counted,
    not guessed at. Returns what was written and how many were not there.
    """
    from strata.catalog import PreparedIndex
    from strata.catalog.types.prepared import relative_key

    index = PreparedIndex.load(data_dir)
    if index is None:
        return None, 0
    by_name = {relative_key(path, data_dir): sample_id for path, sample_id in registered.items()}
    items, missing = [], 0
    for name, entry in index.samples.items():
        if entry.value is None:
            continue
        if name not in by_name:
            missing += 1
            continue
        items.append((by_name[name], entry.value))
    if not items:
        return None, missing
    written = catalog.annotations.annotate_many(label_set_id, items, source="import", batch=batch)
    return written, missing


def labelled_in_index(data_dir: Path) -> int:
    """How many entries of the prepared index beside ``data_dir`` carry a label."""
    from strata.catalog import PreparedIndex

    index = PreparedIndex.load(data_dir)
    if index is None:
        return 0
    return sum(1 for entry in index.samples.values() if entry.value is not None)


def catalogued(catalog, label_set_id: int, collections) -> tuple[int, int]:
    """How many samples are answered or waiting, and how many were skipped."""
    samples = catalog.samples
    dealt = len(samples.unlabelled(label_set_id, collections)) + len(
        samples.labelled(label_set_id, collections)
    )
    return dealt, len(samples.skipped(label_set_id, collections))


def choose_preparer(name: str, produces: str, sample: Path):
    """The preparer class to run: by name, or resolved from what the corpus holds.

    Refuses one whose output this project would not ingest: the corpus
    would convert, and ingest would then admit none of it.
    """
    from strata.catalog.types.preparers import PreparerError, for_source, resolve

    cls = resolve(name) if name else for_source(produces, sample)
    if cls.produces != produces:
        raise PreparerError(
            f"'{cls.name}' produces '{cls.produces}' samples and this project ingests '{produces}'."
        )
    return cls
