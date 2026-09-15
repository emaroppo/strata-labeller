"""Getting a project's files into its catalog: what a directory holds, and registering it.

What ``prepare`` and ``ingest`` do between reading a directory and saying
what happened. Nothing here prints; each step returns what it found and
what it left out. See ``docs/adr/0030`` and ``docs/adr/0036``.
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
    """Everything under ``root``, then checked. See ``docs/adr/0010``."""
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

    In batches, each one transaction, so a re-run after a failure carries
    on (``docs/adr/0032``). A grouping is in the metadata the type records,
    and nothing here treats it apart (``docs/adr/0023``).
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
            metadata_for=lambda p, sources=sources: {
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

    The labels enter with the files, under ``source="import"`` and the batch
    name given (``docs/adr/0028``). A file the index labels but ingest did
    not register is counted. Returns what was written and how many were not
    there.
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

    Refuses one whose output this project would not ingest. See
    ``docs/adr/0033``.
    """
    from strata.catalog.types.preparers import PreparerError, for_source, resolve

    cls = resolve(name) if name else for_source(produces, sample)
    if cls.produces != produces:
        raise PreparerError(
            f"'{cls.name}' produces '{cls.produces}' samples and this project ingests '{produces}'."
        )
    return cls
