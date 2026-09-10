"""A sample type and two preparers that exist only for the prepare tests.

The command under test is the labeller's; the conversions are plugins, and
each real one is tested in its own package. Reaching for one of them here
made the labeller's tests pass only where that plugin happened to be
installed — true in the workspace, and false in the labeller's own
repository. These stand in, registered for the length of a test.

Lives in its own module rather than in a test file because an entry point
names an importable module, and every package's tests directory would
otherwise claim the same one.
"""

import json
from pathlib import Path
from typing import ClassVar

from strata.catalog.builtin_types import Text
from strata.catalog.preparers import Prepared, Preparer
from strata.labels import Span, Spans


class Note(Text):
    """A note's body, stored as a document — the shape a mail preparer produces."""

    segment: ClassVar[str] = "note"


class NotesPreparer(Preparer):
    """JSON lists of notes into documents, each with any spans it arrived with."""

    name: ClassVar[str] = "notes-json"
    produces: ClassVar[str] = "note"
    sources: ClassVar[frozenset[str]] = frozenset({"json"})

    def __init__(self):
        self._empty = 0

    def report(self) -> dict[str, int]:
        return {"skipped_empty": self._empty} if self._empty else {}

    def prepare(self, source: Path, out_dir: Path) -> list[Prepared]:
        written = []
        for note in json.loads(Path(source).read_text()):
            body = note["body"].replace("\r\n", "\n")
            if not body.strip():
                self._empty += 1
                continue
            path = Path(out_dir) / f"{note['id']}.txt"
            path.write_bytes(body.encode())
            spans = [
                Span(labels=[label], start=start, end=end)
                for label, start, end in note.get("entities", [])
            ]
            written.append(
                Prepared(
                    path=path,
                    metadata={"subject": note.get("subject", "")},
                    value=Spans(values=spans) if spans else None,
                )
            )
        return written


class FramesStub(Preparer):
    """Produces another sample type entirely, for the refusal to run it."""

    name: ClassVar[str] = "frames-stub"
    produces: ClassVar[str] = "frames"
    sources: ClassVar[frozenset[str]] = frozenset({"mp4"})

    def prepare(self, source: Path, out_dir: Path) -> list[Prepared]:
        return []
