"""Converting a corpus into the shape a project ingests.

The step that was missing entirely: a project declaring ``type = "frames"``
over a directory of video, or ``type = "email"`` over a directory of mail,
had nothing that could turn one into the other, and ingest admitted none of
it. What is under test here is mostly refusals — the command has to be
unable to leave a corpus half-converted or converted into the wrong thing.

The conversions themselves are plugins, tested in their own packages. These
tests register a stand-in sample type and preparers (``stub_corpus``), so
they pass wherever the labeller is installed, not only where a real plugin
happens to be.
"""

import json
from importlib.metadata import EntryPoint

import pytest
from typer.testing import CliRunner

from strata.catalog.prepared import PreparedIndex
from strata.labeller.cli import app

runner = CliRunner()

NOTES = [
    {
        "id": "0001",
        "subject": "Hello",
        # CRLF on the way in, so canonical form is something done rather
        # than something the fixture happened to have already
        "body": "Dear Bob,\r\n\r\nAda Lovelace wrote this.\r\n",
        "entities": [["PER", 11, 23]],
    }
]


@pytest.fixture(autouse=True)
def stub_plugins(monkeypatch):
    """Register the stand-ins, beside whatever the environment really has."""
    import strata.catalog.preparers as preparers
    import strata.catalog.sample_types as sample_types

    real_types = sample_types._entries()
    monkeypatch.setattr(
        preparers,
        "_entries",
        lambda: [
            EntryPoint("notes-json", "stub_corpus:NotesPreparer", preparers.ENTRY_POINT_GROUP),
            EntryPoint("frames-stub", "stub_corpus:FramesStub", preparers.ENTRY_POINT_GROUP),
        ],
    )
    monkeypatch.setattr(
        sample_types,
        "_entries",
        lambda: [
            *real_types,
            EntryPoint("note", "stub_corpus:Note", sample_types.ENTRY_POINT_GROUP),
        ],
    )


@pytest.fixture
def notes_project(make_project):
    """A span project over notes, with a corpus waiting to be converted."""
    project = make_project(template="text_span", classes=["PER", "ORG"])
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace('type = "text"', 'type = "note"'))

    source = project.root / "data" / "source"
    source.mkdir(parents=True, exist_ok=True)
    (source / "notes.json").write_text(json.dumps(NOTES))
    return project


def prepare(project, *args):
    return runner.invoke(app, ["prepare", "-p", str(project.root), *args])


def test_it_writes_documents_the_project_can_ingest(notes_project):
    result = prepare(notes_project)
    assert result.exit_code == 0, result.output

    documents = list(notes_project.data_dir.glob("*.txt"))
    assert len(documents) == 1
    assert notes_project.sample_type().allows(documents[0])


def test_the_documents_are_already_canonical(notes_project):
    prepare(notes_project)
    [document] = notes_project.data_dir.glob("*.txt")
    data = document.read_bytes()
    # Otherwise ingest rewrites them, and the file on disk and the sample
    # in the catalog stop having the same checksum
    assert b"\r" not in data
    assert notes_project.sample_type().canonicalise(data) == data


def test_the_index_lands_where_ingest_will_read_it(notes_project):
    prepare(notes_project)
    index = PreparedIndex.load(notes_project.data_dir)
    assert index is not None and index.produced_by == "notes-json"
    [entry] = index.samples.values()
    assert entry.metadata["subject"] == "Hello"


def test_candidates_are_reported_but_not_landed(notes_project):
    result = prepare(notes_project)
    # They are guesses. Saying so is the point; storing them without anyone
    # asking would make an export stamp them as answers.
    assert "candidate annotations" in result.output
    index = PreparedIndex.load(notes_project.data_dir)
    [entry] = index.samples.values()
    assert entry.value.values[0].labels == ["PER"]


def test_it_says_where_to_go_next(notes_project):
    assert "ingest" in prepare(notes_project).output


def test_a_missing_corpus_is_an_error_not_an_empty_run(make_project):
    project = make_project(template="text_span", classes=["PER"])
    result = prepare(project)
    assert result.exit_code == 1
    assert "No corpus" in result.output


def test_an_empty_corpus_directory_is_not_a_mistake(notes_project):
    (notes_project.root / "data" / "source" / "notes.json").unlink()
    result = prepare(notes_project)
    # A project exists before its data does, and this is what someone runs
    # to find out
    assert result.exit_code == 0
    assert "No files" in result.output


def test_files_no_conversion_reads_are_reported(notes_project):
    (notes_project.root / "data" / "source" / "notes.rtf").write_text("x")
    result = prepare(notes_project)
    assert "1 file(s) skipped" in result.output


def test_a_preparer_making_the_wrong_thing_is_refused(notes_project):
    # It would convert, and then ingest would admit none of it
    result = prepare(notes_project, "--preparer", "frames-stub")
    assert result.exit_code == 1
    assert "produces 'frames'" in result.output


def test_an_unknown_preparer_names_what_is_installed(notes_project):
    result = prepare(notes_project, "--preparer", "telepathy")
    assert result.exit_code == 1
    assert "notes-json" in result.output


def test_it_can_be_pointed_somewhere_else(notes_project, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "other.json").write_text(json.dumps(NOTES))
    assert prepare(notes_project, "--from", str(elsewhere)).exit_code == 0
    assert list(notes_project.data_dir.glob("*.txt"))


def test_a_second_run_leaves_the_corpus_where_it_was(notes_project):
    prepare(notes_project)
    before = {p.name: p.read_bytes() for p in notes_project.data_dir.glob("*.txt")}
    prepare(notes_project)
    after = {p.name: p.read_bytes() for p in notes_project.data_dir.glob("*.txt")}
    # Every file that comes out identical is a sample that keeps its
    # checksum, and therefore its annotations
    assert after == before


def test_what_was_left_behind_is_said_out_loud(notes_project):
    notes = [*NOTES, {"id": "0002", "body": "   "}]
    (notes_project.root / "data" / "source" / "notes.json").write_text(json.dumps(notes))
    result = prepare(notes_project)
    assert "skipped empty: 1" in result.output
