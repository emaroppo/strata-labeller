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
    (tmp_path / "config.toml").write_text(
        f'[catalog]\nroot = "{tmp_path / "catalog"}"\n'
    )

    def _make(n: int = 6, folder: str = "", kind: str | None = None):
        for i in range(n):
            relative = f"{folder}/img{i:03d}.jpg" if folder else f"img{i:03d}.jpg"
            path = project.data_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"image {i}".encode())
        if kind:
            toml = project.root / "project.toml"
            toml.write_text(toml.read_text().replace('kind = "images"', f'kind = "{kind}"'))
        return project

    return _make


def run(project, *args):
    return runner.invoke(
        app, ["ingest", "-p", str(project.root), "--config", "config.toml", *args]
    )


def catalog_at(tmp_path) -> Catalog:
    return Catalog.local(tmp_path / "catalog")


def test_ingest_creates_the_catalog_and_the_label_set(workspace, tmp_path):
    # The entry point for data has to work against nothing, or there is no
    # way to make the first catalog
    project = workspace(4)
    assert run(project).exit_code == 0

    catalog = catalog_at(tmp_path)
    label_set_id, schema = catalog.label_set(project.name)
    assert schema.classes == ["cat", "dog"]
    assert len(catalog.unlabelled(label_set_id, EVERYTHING)) == 4


def test_ingest_is_safe_to_repeat(workspace, tmp_path):
    project = workspace(5)
    run(project)
    result = run(project)

    catalog = catalog_at(tmp_path)
    label_set_id, _ = catalog.label_set(project.name)
    assert "0 new" in result.stdout
    assert len(catalog.unlabelled(label_set_id, EVERYTHING)) == 5


def test_new_files_are_picked_up_on_a_second_run(workspace, tmp_path):
    project = workspace(3)
    run(project)
    workspace(6)  # three more alongside the originals
    run(project)

    catalog = catalog_at(tmp_path)
    label_set_id, _ = catalog.label_set(project.name)
    assert len(catalog.unlabelled(label_set_id, EVERYTHING)) == 6


def test_frames_are_grouped_by_folder(workspace, tmp_path):
    project = workspace(4, folder="vid1", kind="frames")
    run(project)

    catalog = catalog_at(tmp_path)
    label_set_id, _ = catalog.label_set(project.name)
    assert {s.group_id for s in catalog.unlabelled(label_set_id, EVERYTHING)} == {"vid1"}


def test_plain_images_get_no_group(workspace, tmp_path):
    project = workspace(4)
    run(project)

    catalog = catalog_at(tmp_path)
    label_set_id, _ = catalog.label_set(project.name)
    assert {s.group_id for s in catalog.unlabelled(label_set_id, EVERYTHING)} == {None}


def test_the_source_path_is_recorded(workspace, tmp_path):
    # A blob is addressed by content, so this is the only way back to the
    # file it was read from
    project = workspace(2, folder="vid1", kind="frames")
    run(project)

    catalog = catalog_at(tmp_path)
    label_set_id, _ = catalog.label_set(project.name)
    queue = catalog.unlabelled(label_set_id, EVERYTHING)
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
    label_set_id, _ = catalog.label_set(project.name)
    assert len(catalog.unlabelled(label_set_id, EVERYTHING)) == 3


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
    assert result.exit_code == 0
    assert "No image files" in result.stdout


def test_registering_is_not_queueing(workspace, tmp_path):
    # The catalog holds the whole pool; push sends what is about to be
    # reviewed. That separation is why cataloguing everything costs nothing.
    project = workspace(20)
    run(project)

    catalog = catalog_at(tmp_path)
    label_set_id, _ = catalog.label_set(project.name)
    assert len(catalog.unlabelled(label_set_id, EVERYTHING)) == 20
    assert len(catalog.labelled(label_set_id, EVERYTHING)) == 0
