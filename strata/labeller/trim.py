"""Spans moved off the whitespace around them, against the document they cover.

Label Studio shows a document as HTML, which renders a run of spaces as
one, so a selection made after aligned text comes back starting inside the
run. Export trims what comes in; this finds what is already stored, and
says what it would write before writing it. Nothing here prints.
"""

from dataclasses import dataclass, field

from strata.catalog import Catalog, SampleRow
from strata.contracts import Spans

#: Every origin an answer can have, each trimmed under its own name so the
#: answer keeps its rank. docs/adr/0009
SOURCES = ("human", "import", "model")


def text_of(catalog: Catalog, sample: SampleRow) -> str:
    """A sample's text exactly as stored, which is what span offsets count.

    Decoded from the bytes rather than read as text, which would turn
    ``\\r\\n`` into ``\\n`` and move every offset after the first line.
    """
    return catalog.blobs.get(sample.location).decode("utf-8")


@dataclass(frozen=True)
class Trim:
    """One stored answer, and what it becomes."""

    sample_id: int
    checksum: str
    source: str
    value: Spans
    moved: int
    dropped: int


@dataclass
class TrimReport:
    checked: int = 0
    changes: list[Trim] = field(default_factory=list)
    #: Samples with a disputed answer, left alone: writing to one settles
    #: the dispute, and a trim is not someone looking again.
    disputed: list[str] = field(default_factory=list)

    @property
    def moved(self) -> int:
        return sum(c.moved for c in self.changes)

    @property
    def dropped(self) -> int:
        return sum(c.dropped for c in self.changes)


def plan(catalog: Catalog, label_set_id: int, collections) -> TrimReport:
    """Every current span answer in ``collections`` that trimming would change."""
    report = TrimReport()
    disputed = {d["id"] for d in catalog.conflicts.disputed(label_set_id, collections)}
    for source in SOURCES:
        rows = catalog.samples.labelled(label_set_id, collections, source=source)
        values = catalog.annotations.values_of(label_set_id, [row.id for row in rows])
        for row in rows:
            value = values.get(row.id)
            if not isinstance(value, Spans) or not value.values:
                continue
            report.checked += 1
            trimmed, changed = value.trimmed(text_of(catalog, row))
            if not changed:
                continue
            if row.id in disputed:
                report.disputed.append(row.checksum)
                continue
            dropped = len(value.values) - len(trimmed.values)
            report.changes.append(
                Trim(row.id, row.checksum, source, trimmed, changed - dropped, dropped)
            )
    return report


def apply(catalog: Catalog, label_set_id: int, report: TrimReport) -> int:
    """Write each change as a new answer under its own source. docs/adr/0027

    A new row, never an edit: history keeps what was there, the batch is
    inherited, and frozen versions keep the answer they were frozen with.
    """
    written = 0
    for change in report.changes:
        written += catalog.annotations.annotate(
            change.sample_id, label_set_id, change.value, source=change.source
        )
    return written
