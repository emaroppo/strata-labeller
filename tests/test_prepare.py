"""Converting a corpus into the shape a project ingests.

The step that was missing entirely: a project declaring ``type = "frames"``
over a directory of video, or ``type = "email"`` over a directory of mail,
had nothing that could turn one into the other, and ingest admitted none of
it. What is under test here is mostly refusals — the command has to be
unable to leave a corpus half-converted or converted into the wrong thing.
"""

import json

import pytest
from typer.testing import CliRunner

from strata.catalog.prepared import PreparedIndex
from strata.labeller.cli import app

runner = CliRunner()

CORPUS = [
    {
        "file_path": "mail/0001.eml",
        "messages": [
            {
                "body": "Dear Bob,\r\n\r\nAda Lovelace wrote this.\r\n",
                "headers": {"Subject": "Hello", "Message-ID": "<1@x>"},
                "entities": {"manual": {"PER": [["Ada Lovelace", 13, 25]]}},
            }
        ],
    }
]


@pytest.fixture
def mail_project(make_project):
    """A span project over mail, with a corpus waiting to be converted."""
    project = make_project(template="text_span", classes=["PER", "ORG"])
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace('type = "text"', 'type = "email"'))

    source = project.root / "data" / "source"
    source.mkdir(parents=True, exist_ok=True)
    (source / "corpus.json").write_text(json.dumps(CORPUS))
    return project


def prepare(project, *args):
    return runner.invoke(app, ["prepare", "-p", str(project.root), *args])


def test_it_writes_documents_the_project_can_ingest(mail_project):
    result = prepare(mail_project)
    assert result.exit_code == 0, result.output

    documents = list(mail_project.data_dir.glob("*.txt"))
    assert len(documents) == 1
    sample_type = mail_project.sample_type()
    assert sample_type.allows(documents[0])


def test_the_documents_are_already_canonical(mail_project):
    prepare(mail_project)
    [document] = mail_project.data_dir.glob("*.txt")
    data = document.read_bytes()
    # Otherwise ingest rewrites them, and the file on disk and the sample
    # in the catalog stop having the same checksum
    assert b"\r" not in data
    assert mail_project.sample_type().canonicalise(data) == data


def test_the_index_lands_where_ingest_will_read_it(mail_project):
    prepare(mail_project)
    index = PreparedIndex.load(mail_project.data_dir)
    assert index is not None and index.produced_by == "email-json"
    [entry] = index.samples.values()
    assert entry.metadata["subject"] == "Hello"


def test_candidates_are_reported_but_not_landed(mail_project):
    result = prepare(mail_project)
    # They are a regex's guesses. Saying so is the point; storing them
    # without anyone asking would make an export stamp them as answers.
    assert "candidate annotations" in result.output
    index = PreparedIndex.load(mail_project.data_dir)
    [entry] = index.samples.values()
    assert entry.value.values[0].labels == ["PER"]


def test_it_says_where_to_go_next(mail_project):
    assert "ingest" in prepare(mail_project).output


def test_a_missing_corpus_is_an_error_not_an_empty_run(make_project):
    project = make_project(template="text_span", classes=["PER"])
    result = prepare(project)
    assert result.exit_code == 1
    assert "No corpus" in result.output


def test_an_empty_corpus_directory_is_not_a_mistake(mail_project):
    (mail_project.root / "data" / "source" / "corpus.json").unlink()
    result = prepare(mail_project)
    # A project exists before its data does, and this is what someone runs
    # to find out
    assert result.exit_code == 0
    assert "No files" in result.output


def test_files_no_conversion_reads_are_reported(mail_project):
    (mail_project.root / "data" / "source" / "notes.rtf").write_text("x")
    result = prepare(mail_project)
    assert "1 file(s) skipped" in result.output


def test_a_preparer_making_the_wrong_thing_is_refused(mail_project):
    # It would convert, and then ingest would admit none of it
    result = prepare(mail_project, "--preparer", "video-frames")
    assert result.exit_code == 1
    assert "produces 'frames'" in result.output


def test_an_unknown_preparer_names_what_is_installed(mail_project):
    result = prepare(mail_project, "--preparer", "telepathy")
    assert result.exit_code == 1
    assert "email-json" in result.output


def test_it_can_be_pointed_somewhere_else(mail_project, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "other.json").write_text(json.dumps(CORPUS))
    assert prepare(mail_project, "--from", str(elsewhere)).exit_code == 0
    assert list(mail_project.data_dir.glob("*.txt"))


def test_a_second_run_leaves_the_corpus_where_it_was(mail_project):
    prepare(mail_project)
    before = {p.name: p.read_bytes() for p in mail_project.data_dir.glob("*.txt")}
    prepare(mail_project)
    after = {p.name: p.read_bytes() for p in mail_project.data_dir.glob("*.txt")}
    # Every file that comes out identical is a sample that keeps its
    # checksum, and therefore its annotations
    assert after == before


def test_what_was_left_behind_is_said_out_loud(mail_project):
    corpus = json.loads(json.dumps(CORPUS))
    corpus[0]["messages"].append({"body": "   ", "headers": {}})
    (mail_project.root / "data" / "source" / "corpus.json").write_text(
        json.dumps(corpus)
    )
    result = prepare(mail_project)
    assert "skipped empty: 1" in result.output
