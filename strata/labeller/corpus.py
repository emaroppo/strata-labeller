"""Getting a project's files into its catalog: preparing a source, then registering it.

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


def scan(root: Path, allows: Callable[[Path], bool], ignore: Iterable[Path] = ()) -> Scan:
    """Everything under ``root`` but ``ignore``, then checked. See ``docs/adr/0010``."""
    ignored = set(ignore)
    everything = [p for p in sorted(Path(root).rglob("*")) if p.is_file() and p not in ignored]
    found = [p for p in everything if allows(p)]
    admitted = set(found)
    return Scan(everything, found, [p for p in everything if p not in admitted])


def ingest_files(
    catalog,
    sample_type,
    admission,
    *,
    collections,
    batch: int,
    on_sample: Callable[[Path], None] | None = None,
) -> dict[Path, int]:
    """Register an admitted corpus in ``catalog``; returns each file's sample id.

    ``admission`` is what the catalog checked the prepared index against the
    type into: each file with the metadata to record, ``source_path``
    included. In batches, each one transaction, so a re-run after a failure
    carries on (``docs/adr/0032``). A grouping is in that metadata, and
    nothing here treats it apart (``docs/adr/0023``).
    """
    from strata.catalog import canonical_form

    paths = list(admission.entries)
    canonicalise = canonical_form(type(sample_type))
    registered: dict[Path, int] = {}
    for start in range(0, len(paths), batch):
        chunk = paths[start : start + batch]
        ids = catalog.ingest(
            chunk,
            media=sample_type.media,
            subtype=type(sample_type).subtype(),
            metadata_for=admission.entries.__getitem__,
            canonicalise=canonicalise,
            collections=collections,
            on_sample=on_sample,
        )
        registered.update(zip(chunk, ids, strict=True))
    return registered


def land_labels(catalog, label_set_id: int, admission, registered: dict[Path, int], batch):
    """Store the candidate labels an admitted corpus carries, as one import.

    The labels enter with the files, under ``source="import"`` and the batch
    name given (``docs/adr/0028``). Every labelled file was admitted, so
    every one is registered. Returns what was written, or None where the
    corpus carried no labels.
    """
    items = [(registered[path], value) for path, value in admission.values.items()]
    if not items:
        return None
    return catalog.annotations.annotate_many(label_set_id, items, source="import", batch=batch)


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
    from strata.prepare import PreparerError, preparers

    cls = preparers.resolve(name) if name else preparers.for_source(produces, sample)
    if cls.produces != produces:
        raise PreparerError(
            f"'{cls.name}' produces '{cls.produces}' samples and this project ingests '{produces}'."
        )
    return cls
