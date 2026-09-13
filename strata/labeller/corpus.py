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
) -> int:
    """Register ``found`` in ``catalog`` as ``sample_type`` says; returns the batches written.

    In batches, because a batch is one transaction: a chunk that fails
    leaves the ones before it committed, so a re-run after fixing the file
    carries on. What a sample is grouped by, if anything, is in the metadata
    the type records — a frame's video — and nothing here treats it apart.
    """
    paths = list(found)
    canonicalise = sample_type.canonicalise if type(sample_type).canonicalises() else None
    batches = 0
    for start in range(0, len(paths), batch):
        chunk = paths[start : start + batch]
        sources = {p: str(p.relative_to(data_dir)) for p in chunk}
        catalog.ingest(
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
        batches += 1
    return batches


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
