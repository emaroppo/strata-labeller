"""A scan says what it left out; a preparer is refused when its output would not be ingested."""

import pytest

from strata.catalog.types.preparers import PreparerError
from strata.labeller import corpus


def test_a_scan_keeps_what_is_admitted_and_names_the_kinds_it_skipped(tmp_path):
    for name in ("a.jpg", "b.PNG", "notes.txt", "README"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.jpg").write_bytes(b"x")

    scanned = corpus.scan(tmp_path, lambda p: p.suffix.lower() in {".jpg", ".png"})

    assert [p.name for p in scanned.found] == ["a.jpg", "b.PNG", "c.jpg"]
    assert [p.name for p in scanned.skipped] == ["README", "notes.txt"]
    assert scanned.kinds == ["(none)", ".txt"]
    assert len(scanned.everything) == 5


def test_a_preparer_producing_the_wrong_type_is_refused_by_name(monkeypatch, tmp_path):
    from strata.catalog.types import preparers

    class Frames:
        name = "video-frames"
        produces = "frames"

    monkeypatch.setattr(preparers, "resolve", lambda name: Frames)
    with pytest.raises(PreparerError, match="produces 'frames'.*ingests 'text'"):
        corpus.choose_preparer("video-frames", "text", tmp_path / "x.mp4")
