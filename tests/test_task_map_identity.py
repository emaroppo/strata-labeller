"""Which catalog a cached task map belongs to.

The map is `{sample_id: task_id}`, and sample ids are per-catalog integers.
A map written against one catalog and read against another is not wrong in
any way a computer can see: every id exists on both sides, and every one of
them names a different sample.

What that costs, concretely. `push` uses the map to decide which samples
Label Studio already has, so a foreign map makes it skip real work and
attach predictions to whichever tasks happen to hold those ids. `unskip`
deletes annotations by task id, so a foreign map deletes someone's answers
on unrelated tasks. Neither raises. The symptom is accuracy that stops
improving.

Export and relink are not exposed: both resolve a task through the blob its
URL names, which is content-addressed and means the same thing everywhere.
That is the design the map is a cache of, and it is why a wrong map can
always be rebuilt.
"""

import json

import pytest

from strata.catalog import Catalog
from strata.labeller.labelstudio.sync import (
    TaskMapError,
    load_task_map,
    save_task_map,
    task_map_catalog,
    task_map_path,
)
from strata.labeller.project import Project

A = "20260101T000000-aaaaaaaa"
B = "20260202T000000-bbbbbbbb"


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    (root / "project.toml").write_text(
        '[label_config]\nclasses = ["a"]\n\n[data]\ntype = "image"\n'
    )
    return Project.load(root)


# ----------------------------------------------------------------------
# Round trip
# ----------------------------------------------------------------------


def test_a_map_records_the_catalog_it_was_written_against(project):
    save_task_map(project, 7, {1: 100, 2: 200}, A)
    assert task_map_catalog(project, 7) == A
    assert load_task_map(project, 7, A) == {1: 100, 2: 200}


def test_the_same_catalog_reads_it_back(project):
    save_task_map(project, 7, {1: 100}, A)
    assert load_task_map(project, 7, A) == {1: 100}


def test_a_missing_map_is_empty_not_an_error(project):
    # A project that has never pushed is not a project in trouble
    assert load_task_map(project, 7, A) == {}


# ----------------------------------------------------------------------
# The refusal
# ----------------------------------------------------------------------


def test_another_catalog_is_refused(project):
    save_task_map(project, 7, {1: 100, 2: 200}, A)
    with pytest.raises(TaskMapError):
        load_task_map(project, 7, B)


def test_the_refusal_names_both_catalogs_and_a_way_out(project):
    save_task_map(project, 7, {1: 100}, A)
    with pytest.raises(TaskMapError) as caught:
        load_task_map(project, 7, B)
    message = str(caught.value)
    # Which one it was written against is the actionable half: the fix is
    # either to point the project back, or to rebuild from Label Studio
    assert A in message and B in message
    assert "rebuild-map" in message


def test_ids_that_exist_in_both_are_still_refused(project):
    """The whole point: every id resolves on both sides, to different data.

    There is no check on the ids themselves that could catch this, which is
    why the catalog has to be recorded rather than inferred.
    """
    save_task_map(project, 7, {1: 100, 2: 200, 3: 300}, A)
    with pytest.raises(TaskMapError):
        load_task_map(project, 7, B)


# ----------------------------------------------------------------------
# Maps written before any of this
# ----------------------------------------------------------------------


def test_a_map_from_before_identities_is_adopted(project):
    path = task_map_path(project, 7)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"1": 100, "2": 200}))

    # Adopted rather than refused: it was written by this project against
    # whatever it was pointed at, and refusing would strand every existing
    # queue on an upgrade
    assert task_map_catalog(project, 7) is None
    assert load_task_map(project, 7, A) == {1: 100, 2: 200}


def test_adopting_one_stamps_it_on_the_next_write(project):
    path = task_map_path(project, 7)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"1": 100}))

    mapping = load_task_map(project, 7, A)
    save_task_map(project, 7, mapping, A)

    # So the window in which a map is unguarded closes the first time
    # anything touches it
    assert task_map_catalog(project, 7) == A
    with pytest.raises(TaskMapError):
        load_task_map(project, 7, B)


def test_a_caller_that_names_no_catalog_is_not_blocked(project):
    # Nothing to compare against is not a disagreement. A catalog predating
    # identities has no id to offer, and refusing would break a setup that
    # is working.
    save_task_map(project, 7, {1: 100}, A)
    assert load_task_map(project, 7, None) == {1: 100}


# ----------------------------------------------------------------------
# Through the CLI, on the command that would do damage
# ----------------------------------------------------------------------


def test_unskip_refuses_a_foreign_map(tmp_path, monkeypatch):
    """unskip deletes annotations by task id, straight out of the map."""
    from typer.testing import CliRunner

    from strata.labeller.cli import app
    from strata.labels import ClassificationSchema

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(f'[catalog]\nroot = "{tmp_path / "catalog"}"\n')
    catalog = Catalog.local(tmp_path / "catalog")

    root = tmp_path / "job"
    (root / "data" / "raw").mkdir(parents=True)
    (root / "project.toml").write_text(
        '[label_config]\nclasses = ["a"]\n\n[data]\ntype = "image"\n\n'
        "[label_studio]\nproject_id = 3\n"
    )
    sample = root / "data" / "raw" / "a.jpg"
    sample.write_bytes(b"bytes")
    [sample_id] = catalog.ingest([sample], media="image", collections=["job"])
    label_set_id = catalog.label_sets.create("job", ClassificationSchema(classes=["a"]))
    catalog.annotations.skip(sample_id, label_set_id)

    project = Project.load(root)
    save_task_map(project, 3, {sample_id: 999}, B)

    result = CliRunner().invoke(
        app, ["unskip", "-p", str(root), "--config", "config.toml"]
    )

    assert result.exit_code == 1
    assert B in result.stdout
