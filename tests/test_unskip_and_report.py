"""Returning skipped samples, and reading the training history.

Both used to work on files the catalog replaced — dataset.json and
rounds/*/metadata.json — and both are mostly about not asserting more than
the record supports.
"""

import pytest
from typer.testing import CliRunner

from strata.catalog import Catalog
from strata.labeller.cli import app
from strata.labels import Choices, ClassificationSchema
from strata.modelling import Run, RunStore

runner = CliRunner()


@pytest.fixture
def workspace(project, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(f'[catalog]\nroot = "{tmp_path / "catalog"}"\n')
    catalog = Catalog.local(tmp_path / "catalog")
    label_set_id = catalog.create_label_set(
        project.name, ClassificationSchema(classes=["cat", "dog"])
    )
    paths = []
    for i in range(6):
        path = project.data_dir / f"img{i}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"image {i}".encode())
        paths.append(path)
    ids = catalog.ingest(paths, media="image")
    return project, catalog, label_set_id, ids


def run_cmd(project, *args):
    return runner.invoke(
        app, [*args, "-p", str(project.root), "--config", "config.toml"]
    )


def report_cmd(project, *args):
    # report reads the project's own run store; it needs no host config
    return runner.invoke(app, ["report", "-p", str(project.root), *args])


# ----------------------------------------------------------------------
# unskip
# ----------------------------------------------------------------------


def test_unskip_returns_samples_to_the_queue(workspace):
    project, catalog, label_set_id, ids = workspace
    catalog.skip(ids[0], label_set_id)
    catalog.skip(ids[1], label_set_id)

    assert run_cmd(project, "unskip").exit_code == 0
    # The queue is the absence of a row, so the row is deleted rather than
    # flagged
    assert catalog.skipped(label_set_id) == []
    assert len(catalog.unlabelled(label_set_id)) == 6


def test_unskip_honours_a_limit(workspace):
    project, catalog, label_set_id, ids = workspace
    for i in range(4):
        catalog.skip(ids[i], label_set_id)

    run_cmd(project, "unskip", "--limit", "2")
    assert len(catalog.skipped(label_set_id)) == 2


def test_unskip_leaves_annotations_alone(workspace):
    project, catalog, label_set_id, ids = workspace
    catalog.annotate(ids[0], label_set_id, Choices(values=["cat"]))
    catalog.skip(ids[1], label_set_id)

    run_cmd(project, "unskip")
    # A skip is the only state this undoes; discarding an answer by accident
    # would be far worse
    assert catalog.annotation_of(ids[0], label_set_id) == Choices(values=["cat"])
    assert len(catalog.labelled(label_set_id)) == 1


def test_unskip_with_nothing_skipped_says_so(workspace):
    project, _, _, _ = workspace
    result = run_cmd(project, "unskip")
    assert result.exit_code == 0
    assert "Nothing is skipped" in result.stdout


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------


def a_run(store, **overrides) -> Run:
    base = dict(
        id=0,
        dataset="demo",
        dataset_version=1,
        label_set="demo",
        model="toy",
        model_version="1",
        classes=["cat"],
    )
    metrics = overrides.pop("metrics", {"val_accuracy": 0.5})
    return store.record(Run(**{**base, **overrides}), metrics)


@pytest.fixture
def runs(project, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(f'[catalog]\nroot = "{tmp_path / "catalog"}"\n')
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text() + '\n[catalog]\ndataset = "demo"\n')
    from strata.labeller.project import Project

    return Project.load(project.root), RunStore.local(project.runs_dir)


def test_report_needs_a_run_store(project, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(f'[catalog]\nroot = "{tmp_path / "catalog"}"\n')
    result = report_cmd(project)
    assert result.exit_code == 1
    assert "No runs recorded" in result.stdout


def test_report_lists_the_history(runs):
    project, store = runs
    a_run(store, dataset_version=1, metrics={"val_accuracy": 0.80})
    a_run(store, dataset_version=2, parent_run_id=1, metrics={"val_accuracy": 0.85})
    result = report_cmd(project)
    assert result.exit_code == 0
    assert "0.8000" in result.stdout and "0.8500" in result.stdout


def test_a_change_is_shown_only_against_a_run_s_own_parent(runs):
    project, store = runs
    a_run(store, dataset_version=1, metrics={"val_accuracy": 0.80})
    a_run(store, dataset_version=2, parent_run_id=1, metrics={"val_accuracy": 0.85})
    a_run(store, dataset_version=2, metrics={"val_accuracy": 0.60})  # cold
    result = report_cmd(project)
    # One delta, for the one pair that continues each other
    assert result.stdout.count("+0.0500") == 1
    assert "unchained" in result.stdout


def test_no_change_across_a_version_going_backwards(runs):
    project, store = runs
    # An imported round, then the first trained on the catalog: the lineage
    # is real but the two were scored on different held-out samples
    a_run(store, dataset_version=33, metrics={"val_accuracy": 0.877})
    a_run(store, dataset_version=2, parent_run_id=1, metrics={"val_accuracy": 0.978})
    result = report_cmd(project)
    assert "+0.1010" not in result.stdout
    assert "from 1" in result.stdout


def test_report_details_one_run(runs):
    project, store = runs
    a_run(store, dataset_version=1, metrics={"val_accuracy": 0.9, "loss": 0.1})
    result = report_cmd(project, "--run", "1")
    assert result.exit_code == 0
    assert "unchained" in result.stdout
    assert "0.9000" in result.stdout


def test_detailing_a_run_shows_its_chain(runs):
    project, store = runs
    a_run(store, dataset_version=1)
    a_run(store, dataset_version=2, parent_run_id=1)
    result = report_cmd(project, "--run", "2")
    assert "1 -> 2" in result.stdout


def test_an_unknown_metric_suggests_another(runs):
    project, store = runs
    a_run(store, metrics={"accuracy": 0.9})
    result = report_cmd(project, "--metric", "f1")
    assert result.exit_code == 1
    assert "--metric accuracy" in result.stdout
