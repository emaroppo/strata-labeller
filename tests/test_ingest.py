"""Registering files in the catalog.

The entry point for data, so it has to work against nothing: no catalog, no
label set, no prior run. And it has to be safe to repeat, because the way it
is used is to point it at a growing directory.
"""

import pytest
from typer.testing import CliRunner

from strata.catalog import EVERYTHING, Catalog
from strata.labeller.cli import app

runner = CliRunner()


@pytest.fixture
def workspace(project, tmp_path, monkeypatch):
    """A project with files on disk and a config naming a catalog."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(f'[catalog]\nroot = "{tmp_path / "catalog"}"\n')

    def _make(n: int = 6, folder: str = "", sample_type: str | None = None):
        for i in range(n):
            relative = f"{folder}/img{i:03d}.jpg" if folder else f"img{i:03d}.jpg"
            path = project.data_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"image {i}".encode())
        if sample_type:
            toml = project.root / "project.toml"
            toml.write_text(toml.read_text().replace('type = "image"', f'type = "{sample_type}"'))
        return project

    return _make


def run(project, *args):
    return runner.invoke(app, ["ingest", "-p", str(project.root), "--config", "config.toml", *args])


def catalog_at(tmp_path) -> Catalog:
    return Catalog.local(tmp_path / "catalog")


def test_ingest_creates_the_catalog_and_the_label_set(workspace, tmp_path):
    # The entry point for data has to work against nothing, or there is no
    # way to make the first catalog
    project = workspace(4)
    assert run(project).exit_code == 0

    catalog = catalog_at(tmp_path)
    label_set_id, schema = catalog.label_sets.get(project.name)
    assert schema.classes == ["cat", "dog"]
    assert len(catalog.samples.unlabelled(label_set_id, EVERYTHING)) == 4


def test_ingest_is_safe_to_repeat(workspace, tmp_path):
    project = workspace(5)
    run(project)
    result = run(project)

    catalog = catalog_at(tmp_path)
    label_set_id, _ = catalog.label_sets.get(project.name)
    assert "0 new" in result.stdout
    assert len(catalog.samples.unlabelled(label_set_id, EVERYTHING)) == 5


def test_new_files_are_picked_up_on_a_second_run(workspace, tmp_path):
    project = workspace(3)
    run(project)
    workspace(6)  # three more alongside the originals
    run(project)

    catalog = catalog_at(tmp_path)
    label_set_id, _ = catalog.label_sets.get(project.name)
    assert len(catalog.samples.unlabelled(label_set_id, EVERYTHING)) == 6


def test_frames_record_their_folder_as_their_video(workspace, tmp_path):
    project = workspace(4, folder="vid1", sample_type="frames")
    run(project)

    catalog = catalog_at(tmp_path)
    label_set_id, _ = catalog.label_sets.get(project.name)
    rows = catalog.samples.unlabelled(label_set_id, EVERYTHING)
    assert {s.metadata["video"] for s in rows} == {"vid1"}


def test_plain_images_record_no_grouping(workspace, tmp_path):
    project = workspace(4)
    run(project)

    catalog = catalog_at(tmp_path)
    label_set_id, _ = catalog.label_sets.get(project.name)
    rows = catalog.samples.unlabelled(label_set_id, EVERYTHING)
    assert all("video" not in s.metadata for s in rows)


def test_the_source_path_is_recorded(workspace, tmp_path):
    # A blob is addressed by content, so this is the only way back to the
    # file it was read from
    project = workspace(2, folder="vid1", sample_type="frames")
    run(project)

    catalog = catalog_at(tmp_path)
    label_set_id, _ = catalog.label_sets.get(project.name)
    queue = catalog.samples.unlabelled(label_set_id, EVERYTHING)
    assert {(s.metadata or {}).get("source_path") for s in queue} == {
        "vid1/img000.jpg",
        "vid1/img001.jpg",
    }


def test_files_of_another_media_type_are_ignored(workspace, tmp_path):
    project = workspace(3)
    (project.data_dir / "notes.txt").write_text("not an image")
    (project.data_dir / "clip.mp4").write_bytes(b"not an image either")
    run(project)

    catalog = catalog_at(tmp_path)
    label_set_id, _ = catalog.label_sets.get(project.name)
    assert len(catalog.samples.unlabelled(label_set_id, EVERYTHING)) == 3


def test_a_missing_data_root_is_an_error(project, tmp_path, monkeypatch):
    import shutil

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(f'[catalog]\nroot = "{tmp_path / "catalog"}"\n')
    shutil.rmtree(project.data_dir)
    result = run(project)
    assert result.exit_code == 1
    assert "Data root does not exist" in result.stdout


def test_an_empty_data_root_says_so_rather_than_failing(workspace, tmp_path):
    project = workspace(0)
    result = run(project)
    # A project exists before its data does, and this is what someone runs
    # to find out whether it has arrived
    assert result.exit_code == 0
    assert "No files" in result.stdout


def test_files_that_are_all_the_wrong_kind_are_an_error(workspace, tmp_path):
    project = workspace(0)
    for name in ("notes.md", "readme.txt"):
        (project.data_dir / name).write_text("x")

    result = run(project)

    # Files, and none admitted: the wrong folder or the wrong type. Saying
    # nothing here reports an empty corpus as a success.
    assert result.exit_code == 1
    assert "None of the 2" in result.stdout


def test_registering_is_not_queueing(workspace, tmp_path):
    # The catalog holds the whole pool; push sends what is about to be
    # reviewed. That separation is why cataloguing everything costs nothing.
    project = workspace(20)
    run(project)

    catalog = catalog_at(tmp_path)
    label_set_id, _ = catalog.label_sets.get(project.name)
    assert len(catalog.samples.unlabelled(label_set_id, EVERYTHING)) == 20
    assert len(catalog.samples.labelled(label_set_id, EVERYTHING)) == 0


def test_labels_the_corpus_arrived_with_land_at_ingest(make_project, tmp_path):
    """A prepared corpus is labelled the moment it is catalogued.

    The index a preparer left beside the files is what ingest already reads
    for metadata; the labels in it enter with the files, as an import batch
    a person can spot-review later.
    """
    from strata.catalog import PreparedIndex, PreparedSample
    from strata.labels import Choices

    project = make_project("demo", classes=["cat", "dog"])
    (tmp_path / "config.toml").write_text(f'[catalog]\nroot = "{tmp_path / "catalog"}"\n')
    project.data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (project.data_dir / f"img{i:03d}.jpg").write_bytes(f"image {i}".encode())
    PreparedIndex(
        samples={
            "img000.jpg": PreparedSample(value=Choices(values=["cat"])),
            "img001.jpg": PreparedSample(value=Choices(values=["dog"])),
            "img002.jpg": PreparedSample(),
        }
    ).save(project.data_dir)

    # The labels are part of what the corpus is, so landing them wants a name
    result = run(project)
    assert result.exit_code == 1
    assert "--import" in result.output

    result = run(project, "--import", "pv-1")
    assert result.exit_code == 0, result.output
    assert "2 label(s) landed as import 'pv-1'" in result.stdout

    catalog = catalog_at(tmp_path)
    label_set_id, _ = catalog.label_sets.get(project.name)
    labelled = catalog.samples.labelled(label_set_id, EVERYTHING)
    assert len(labelled) == 2
    assert {s.id for s in catalog.samples.unreviewed(label_set_id, EVERYTHING)} == {
        s.id for s in labelled
    }
    [answer] = catalog.annotations.history(labelled[0].id, label_set_id)
    assert (answer.source, answer.batch) == ("import", "pv-1")
    assert len(catalog.samples.unlabelled(label_set_id, EVERYTHING)) == 1
