"""Returning skipped samples, and reading the training history.

Both used to work on files the catalog replaced — dataset.json and
rounds/*/metadata.json — and both are mostly about not asserting more than
the record supports.
"""

import json

import pytest
from typer.testing import CliRunner

from strata.catalog import EVERYTHING, Catalog
from strata.labeller.cli import app
from strata.labels import Choices, ClassificationSchema
from strata.modelling import Run, RunStore

runner = CliRunner()


@pytest.fixture
def workspace(project, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(f'[catalog]\nroot = "{tmp_path / "catalog"}"\n')
    catalog = Catalog.local(tmp_path / "catalog")
    label_set_id = catalog.label_sets.create(
        project.name, ClassificationSchema(classes=["cat", "dog"])
    )
    paths = []
    for i in range(6):
        path = project.data_dir / f"img{i}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"image {i}".encode())
        paths.append(path)
    # Tagged into the collection the project draws from: a sample in none is
    # invisible to every project, which is the point of the scoping
    ids = catalog.ingest(paths, media="image", collections=project.collections)
    return project, catalog, label_set_id, ids


def run_cmd(project, *args):
    return runner.invoke(app, [*args, "-p", str(project.root), "--config", "config.toml"])


def report_cmd(project, *args):
    # The run store is the project's own; the host config names the catalog
    # whose history it is, and config.toml in the working directory is it
    return runner.invoke(app, ["report", "-p", str(project.root), *args])


# ----------------------------------------------------------------------
# unskip
# ----------------------------------------------------------------------


def test_unskip_returns_samples_to_the_queue(workspace):
    project, catalog, label_set_id, ids = workspace
    catalog.annotations.skip(ids[0], label_set_id)
    catalog.annotations.skip(ids[1], label_set_id)

    assert run_cmd(project, "unskip").exit_code == 0
    # The queue is the absence of a row, so the row is deleted rather than
    # flagged
    assert catalog.samples.skipped(label_set_id, EVERYTHING) == []
    assert len(catalog.samples.unlabelled(label_set_id, EVERYTHING)) == 6


def test_unskip_honours_a_limit(workspace):
    project, catalog, label_set_id, ids = workspace
    for i in range(4):
        catalog.annotations.skip(ids[i], label_set_id)

    run_cmd(project, "unskip", "--limit", "2")
    assert len(catalog.samples.skipped(label_set_id, EVERYTHING)) == 2


def test_unskip_leaves_annotations_alone(workspace):
    project, catalog, label_set_id, ids = workspace
    catalog.annotations.annotate(ids[0], label_set_id, Choices(values=["cat"]))
    catalog.annotations.skip(ids[1], label_set_id)

    run_cmd(project, "unskip")
    # A skip is the only state this undoes; discarding an answer by accident
    # would be far worse
    assert catalog.annotations.annotation_of(ids[0], label_set_id) == Choices(values=["cat"])
    assert len(catalog.samples.labelled(label_set_id, EVERYTHING)) == 1


def test_unskip_with_nothing_skipped_says_so(workspace):
    project, _, _, _ = workspace
    result = run_cmd(project, "unskip")
    assert result.exit_code == 0
    assert "Nothing is skipped" in result.stdout


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------


def a_run(store, **overrides) -> Run:
    base = {
        "id": "",
        "dataset": "demo",
        "dataset_version": 1,
        "label_set": "demo",
        "model": "toy",
        "model_version": "1",
        "classes": ["cat"],
    }
    metrics = overrides.pop("metrics", {"val_accuracy": 0.5})
    return store.record(Run(**{**base, **overrides}), metrics)


@pytest.fixture
def runs(project, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(f'[catalog]\nroot = "{tmp_path / "catalog"}"\n')
    toml = project.root / "project.toml"
    # The scaffold writes a [catalog] section; the dataset name goes in it
    toml.write_text(toml.read_text().replace("[catalog]\n", '[catalog]\ndataset = "demo"\n'))
    from strata.catalog import Catalog
    from strata.labeller.project import LabellingProject

    Catalog.local(tmp_path / "catalog")
    return LabellingProject.load(project.root), RunStore.local(project.runs_dir)


def test_report_needs_a_run_store(project, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(f'[catalog]\nroot = "{tmp_path / "catalog"}"\n')
    result = report_cmd(project)
    assert result.exit_code == 1
    assert "No runs recorded" in result.stdout


def test_report_shows_the_projects_catalogs_history_only(runs, tmp_path):
    from strata.catalog import Catalog

    project, store = runs
    own = Catalog.local(tmp_path / "catalog").id
    # Three runs over one dataset name: this catalog's, one from before
    # catalogs had identities, and one from a catalog this project no
    # longer names. A delta across the last would measure nothing.
    a_run(store, catalog_id=own, metrics={"val_accuracy": 0.70})
    a_run(store, catalog_id=None, metrics={"val_accuracy": 0.75})
    elsewhere = a_run(store, catalog_id="other-catalog", metrics={"val_accuracy": 0.99})

    payload = _json_of(report_cmd(project, "--json"))
    shown = [r["id"] for r in payload["runs"]]
    assert len(shown) == 2 and elsewhere.id not in shown


def test_report_says_how_imported_labels_fared_under_review(runs, tmp_path):
    from strata.catalog import Catalog
    from strata.labels import Choices, ClassificationSchema

    project, store = runs
    a_run(store)
    catalog = Catalog.local(tmp_path / "catalog")
    raw = tmp_path / "raw"
    raw.mkdir()
    paths = [raw / f"{i}.jpg" for i in range(3)]
    for i, path in enumerate(paths):
        path.write_bytes(f"image {i}".encode())
    ids = catalog.ingest(paths, media="image")
    label_set = catalog.label_sets.create(
        project.label_set_name, ClassificationSchema(classes=["cat", "dog"])
    )
    catalog.annotations.annotate_many(
        label_set, [(i, Choices(values=["cat"])) for i in ids], source="import", batch="pv"
    )
    # One confirmed, one corrected, one nobody looked at
    catalog.annotations.annotate(ids[0], label_set, Choices(values=["cat"]))
    catalog.annotations.annotate(ids[1], label_set, Choices(values=["dog"]))

    payload = _json_of(report_cmd(project, "--json"))
    assert payload["imports"] == {"pv": {"accepted": 1, "corrected": 1, "pending": 1}}
    out = report_cmd(project).stdout
    assert "Imported labels under review" in out


def test_report_lists_the_history(runs):
    project, store = runs
    first = a_run(store, dataset_version=1, metrics={"val_accuracy": 0.80})
    a_run(store, dataset_version=2, parent_run_id=first.id, metrics={"val_accuracy": 0.85})
    result = report_cmd(project)
    assert result.exit_code == 0
    assert "0.8000" in result.stdout and "0.8500" in result.stdout


def test_a_change_is_shown_only_against_a_run_s_own_parent(runs):
    project, store = runs
    first = a_run(store, dataset_version=1, metrics={"val_accuracy": 0.80})
    a_run(store, dataset_version=2, parent_run_id=first.id, metrics={"val_accuracy": 0.85})
    a_run(store, dataset_version=2, metrics={"val_accuracy": 0.60})  # cold
    result = report_cmd(project)
    # One delta, for the one pair that continues each other
    assert result.stdout.count("+0.0500") == 1
    assert "unchained" in result.stdout


def test_no_change_across_a_version_going_backwards(runs):
    project, store = runs
    # An imported round, then the first trained on the catalog: the lineage
    # is real but the two were scored on different held-out samples
    first = a_run(store, dataset_version=33, metrics={"val_accuracy": 0.877})
    a_run(store, dataset_version=2, parent_run_id=first.id, metrics={"val_accuracy": 0.978})
    result = report_cmd(project)
    assert "+0.1010" not in result.stdout
    assert "warm" in result.stdout


def test_report_details_one_run(runs):
    project, store = runs
    run = a_run(store, dataset_version=1, metrics={"val_accuracy": 0.9, "loss": 0.1})
    result = report_cmd(project, "--run", run.id)
    assert result.exit_code == 0
    assert "unchained" in result.stdout
    assert "0.9000" in result.stdout


def test_detailing_a_run_shows_its_chain(runs):
    project, store = runs
    first = a_run(store, dataset_version=1)
    second = a_run(store, dataset_version=2, parent_run_id=first.id)
    result = report_cmd(project, "--run", second.id)
    assert f"{first.short} -> {second.short}" in result.stdout


def test_an_unknown_metric_suggests_another(runs):
    project, store = runs
    a_run(store, metrics={"accuracy": 0.9})
    result = report_cmd(project, "--metric", "f1")
    assert result.exit_code == 1
    assert "--metric accuracy" in result.stdout


# ----------------------------------------------------------------------
# report --json
# ----------------------------------------------------------------------


def _json_of(result):
    """The payload, and proof there is nothing else on stdout."""
    return json.loads(result.stdout)


def test_report_json_carries_the_history(runs):
    project, store = runs
    first = a_run(store, dataset_version=1, metrics={"val_accuracy": 0.80})
    a_run(store, dataset_version=2, parent_run_id=first.id, metrics={"val_accuracy": 0.85})

    payload = _json_of(report_cmd(project, "--json"))

    assert payload["dataset"] == "demo"
    assert payload["metric"] == "val_accuracy"
    assert [r["value"] for r in payload["runs"]] == [0.80, 0.85]


def test_report_json_is_the_only_thing_on_stdout(runs):
    """It has to pipe. A table drawn alongside it would be a parse error."""
    project, store = runs
    a_run(store, dataset_version=1)
    result = report_cmd(project, "--json")
    assert result.stdout.lstrip().startswith("{")
    json.loads(result.stdout)


def test_report_json_says_null_where_a_delta_would_mean_nothing(runs):
    """Absent is not zero: 'not comparable' and 'did not move' differ."""
    project, store = runs
    first = a_run(store, dataset_version=1, metrics={"val_accuracy": 0.80})
    a_run(store, dataset_version=2, parent_run_id=first.id, metrics={"val_accuracy": 0.85})
    a_run(store, dataset_version=2, metrics={"val_accuracy": 0.60})  # cold

    runs_out = _json_of(report_cmd(project, "--json"))["runs"]
    deltas = [r["delta"] for r in runs_out]

    assert deltas[0] is None  # nothing before it
    assert deltas[1] == pytest.approx(0.05)
    assert deltas[2] is None  # a cold run continues nothing
    assert [r["lineage"] for r in runs_out] == ["unchained", "warm", "unchained"]


def test_report_json_carries_params_and_classes(runs):
    """What lets a consumer decide two runs are not asking one question.

    A metric can move because the data changed, the model changed, or what
    the model was told to do changed. Only the last is invisible in a table
    of numbers, and these two fields are where it shows.
    """
    project, store = runs
    a_run(store, dataset_version=1, params={"arm": "flat"}, classes=["cat", "dog"])

    run = _json_of(report_cmd(project, "--json"))["runs"][0]

    assert run["params"] == {"arm": "flat"}
    assert run["classes"] == ["cat", "dog"]


def test_report_json_details_one_run_with_its_chain(runs):
    project, store = runs
    first = a_run(store, dataset_version=1)
    second = a_run(
        store, dataset_version=2, parent_run_id=first.id, metrics={"val_accuracy": 0.9, "loss": 0.1}
    )

    payload = _json_of(report_cmd(project, "--run", second.id, "--json"))

    assert payload["run"]["id"] == second.id
    assert payload["run"]["metrics"] == {"val_accuracy": 0.9, "loss": 0.1}
    assert [r["id"] for r in payload["chain"]] == [first.id, second.id]


def test_report_shows_how_much_of_the_validation_set_nobody_checked(runs):
    from strata.modelling import Seen

    project, store = runs
    run = store.record(
        Run(
            id="",
            dataset="demo",
            dataset_version=1,
            label_set="demo",
            model="toy",
            model_version="1",
            classes=["cat"],
        ),
        {"val_accuracy": 0.9},
        saw=[
            Seen("a" * 64, "train", "wave-1", True),
            Seen("b" * 64, "val", "wave-1", False),
            Seen("c" * 64, "val", "wave-1", True),
            Seen("d" * 64, "val", "wave-2", False),
        ],
    )
    result = report_cmd(project)
    assert result.exit_code == 0, result.stdout
    # Beside the metric: two of the three validation samples were never checked
    assert "2/3" in result.stdout

    detail = report_cmd(project, "--run", run.id)
    assert "nobody checked" in detail.stdout
    assert "wave-2" in detail.stdout

    payload = json.loads(report_cmd(project, "--json").stdout)
    assert payload["runs"][0]["unchecked"] == [
        {"side": "train", "batch": "wave-1", "samples": 1, "unreviewed": 0},
        {"side": "val", "batch": "wave-1", "samples": 2, "unreviewed": 1},
        {"side": "val", "batch": "wave-2", "samples": 1, "unreviewed": 1},
    ]


def test_a_run_that_did_not_say_shows_a_dash_not_a_zero(runs):
    project, store = runs
    a_run(store, metrics={"val_accuracy": 0.5})
    result = report_cmd(project)
    assert result.exit_code == 0
    assert "—" in result.stdout and "0/0" not in result.stdout
