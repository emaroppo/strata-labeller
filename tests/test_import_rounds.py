"""Carrying pre-catalog rounds into the run store.

What matters is that the metric curve survives intact, that nothing is
invented about lineage, and that a round whose checkpoint has gone is
recorded rather than dropped — the numbers are worth keeping even when the
weights are not there.
"""

import json

import pytest

from strata.labeller.import_rounds import describe, import_rounds, read_rounds
from strata.modelling import RunStore


@pytest.fixture
def with_rounds(project):
    """Write rounds/round_NNN/metadata.json the way train.py did."""

    def _make(n: int = 3, classes=("cat", "dog"), checkpoints: bool = True, **overrides):
        for i in range(1, n + 1):
            directory = project.rounds_dir / f"round_{i:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            relative = f"checkpoints/round_{i:03d}.pt"
            if checkpoints:
                path = project.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"weights")
            payload = {
                "round": i,
                "timestamp": f"2026-01-0{i}T00:00:00",
                "num_train": 40 + i,
                "num_val": 10,
                "num_unlabeled": 100 - i,
                "num_skipped": 0,
                "schema": "image_classification",
                "classes": list(classes),
                "metrics": {"accuracy": 0.5 + i / 10, "val_accuracy": 0.4 + i / 10},
                "checkpoint": relative,
            }
            payload.update(overrides)
            (directory / "metadata.json").write_text(json.dumps(payload))
        return project

    return _make


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------


def test_rounds_are_read_oldest_first(with_rounds):
    project = with_rounds(3)
    assert [m["round"] for m in read_rounds(project)] == [1, 2, 3]


def test_a_project_with_no_rounds_reads_empty(project):
    assert read_rounds(project) == []


def test_round_order_does_not_depend_on_directory_order(with_rounds):
    project = with_rounds(2)
    # A round numbered out of step with its directory name still sorts on
    # the number, which is what the curve is indexed by
    directory = project.rounds_dir / "round_002"
    payload = json.loads((directory / "metadata.json").read_text())
    payload["round"] = 0
    (directory / "metadata.json").write_text(json.dumps(payload))
    assert [m["round"] for m in read_rounds(project)] == [0, 1]


# ----------------------------------------------------------------------
# Importing
# ----------------------------------------------------------------------


def test_every_round_becomes_a_run(with_rounds):
    project = with_rounds(4)
    assert import_rounds(project).imported == 4


def test_the_metric_curve_survives(with_rounds, tmp_path):
    project = with_rounds(3)
    store = RunStore.local(tmp_path / "runs")
    import_rounds(project, store)

    # The thing worth importing: accuracy per round, as one query
    curve = [round(v, 2) for _, _, v in store.history(project.dataset_name, "accuracy")]
    assert curve == [0.6, 0.7, 0.8]


def test_both_metrics_come_across(with_rounds, tmp_path):
    project = with_rounds(1)
    store = RunStore.local(tmp_path / "runs")
    [run] = import_rounds(project, store).runs
    assert set(store.get(run.id).metrics) == {"accuracy", "val_accuracy"}


def test_non_numeric_metrics_are_left_behind(with_rounds, tmp_path):
    project = with_rounds(1, metrics={"accuracy": 0.9, "note": "looked fine"})
    store = RunStore.local(tmp_path / "runs")
    [run] = import_rounds(project, store).runs
    assert set(store.get(run.id).metrics) == {"accuracy"}


def test_the_class_list_is_carried(with_rounds, tmp_path):
    project = with_rounds(1, classes=("cat", "dog", "bird"))
    store = RunStore.local(tmp_path / "runs")
    [run] = import_rounds(project, store).runs
    # Output neurons map to this by position, so a warm start needs it
    assert store.get(run.id).classes == ["cat", "dog", "bird"]


def test_the_checkpoint_is_referenced_where_it_sits(with_rounds, tmp_path):
    project = with_rounds(1)
    store = RunStore.local(tmp_path / "runs")
    [run] = import_rounds(project, store).runs
    # Referenced rather than copied: checkpoints are the large part of a
    # project and duplicating them to rename them is a poor trade
    assert run.checkpoint == project.root / "checkpoints" / "round_001.pt"
    assert run.checkpoint.exists()


def test_an_imported_round_records_no_dataset_version(with_rounds, tmp_path):
    project = with_rounds(2)
    store = RunStore.local(tmp_path / "runs")
    runs = import_rounds(project, store).runs
    # It trained on samples no dataset version describes, and standing a
    # round number in for one made it read as sharing data with a catalog
    # dataset that happened to carry the same number
    assert [r.dataset_version for r in runs] == [None, None]


def test_rounds_are_imported_in_order(with_rounds, tmp_path):
    # A round number was never a run id — it lined up only because ids
    # autoincremented. What has to survive is the order they happened in.
    project = with_rounds(3)
    store = RunStore.local(tmp_path / "runs")

    ids = [r.id for r in import_rounds(project, store).runs]

    assert len(ids) == 3
    assert ids == sorted(ids)
    assert len(set(ids)) == 3


# ----------------------------------------------------------------------
# Lineage
# ----------------------------------------------------------------------


def test_imported_runs_are_unchained_by_default(with_rounds, tmp_path):
    project = with_rounds(3)
    store = RunStore.local(tmp_path / "runs")
    # Warm starting arrived partway through, and nothing on disk says which
    # rounds were cold. Asserting a chain would be a claim about the metrics
    # that the files do not support.
    assert all(r.parent_run_id is None for r in import_rounds(project, store).runs)


def test_chain_links_them_in_order(with_rounds, tmp_path):
    project = with_rounds(3)
    store = RunStore.local(tmp_path / "runs")
    runs = import_rounds(project, store, chain=True).runs
    assert [r.parent_run_id for r in runs] == [None, runs[0].id, runs[1].id]


def test_a_chained_import_walks_back_to_the_first(with_rounds, tmp_path):
    project = with_rounds(3)
    store = RunStore.local(tmp_path / "runs")
    runs = import_rounds(project, store, chain=True).runs
    assert [r.id for r in store.chain(runs[-1].id)] == [r.id for r in runs]


# ----------------------------------------------------------------------
# What cannot be carried
# ----------------------------------------------------------------------


def test_a_round_without_classes_is_skipped(with_rounds, tmp_path):
    project = with_rounds(2, classes=())
    report = import_rounds(project, RunStore.local(tmp_path / "runs"))
    # Without the class list the checkpoint's outputs match nothing, so the
    # run could never be continued from
    assert report.imported == 0
    assert len(report.skipped) == 2


def test_a_missing_checkpoint_is_recorded_without_one(with_rounds, tmp_path):
    project = with_rounds(2, checkpoints=False)
    report = import_rounds(project, RunStore.local(tmp_path / "runs"))
    # The numbers are worth keeping even when the weights are gone
    assert report.imported == 2
    assert len(report.missing_checkpoints) == 2
    assert all(r.checkpoint is None for r in report.runs)


def test_importing_twice_duplicates(with_rounds, tmp_path):
    # Stated rather than guarded: the run store is append-only and a run is
    # an event, so a second import is a second set of events
    project = with_rounds(2)
    store = RunStore.local(tmp_path / "runs")
    import_rounds(project, store)
    import_rounds(project, store)
    assert len(store.history(project.dataset_name, "accuracy")) == 4


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def test_the_summary_says_lineage_was_not_claimed(with_rounds, tmp_path):
    project = with_rounds(2)
    report = import_rounds(project, RunStore.local(tmp_path / "runs"))
    assert "unchained" in "\n".join(describe(report, chained=False))


def test_a_chained_summary_makes_no_such_note(with_rounds, tmp_path):
    project = with_rounds(2)
    report = import_rounds(project, RunStore.local(tmp_path / "runs"), chain=True)
    assert "unchained" not in "\n".join(describe(report, chained=True))


def test_the_summary_names_missing_checkpoints(with_rounds, tmp_path):
    project = with_rounds(1, checkpoints=False)
    report = import_rounds(project, RunStore.local(tmp_path / "runs"))
    assert "no checkpoint" in "\n".join(describe(report, chained=False))
