"""The delta rule: a change is shown only against a run's own parent, on comparable data."""

from types import SimpleNamespace

import pytest

from strata.labeller import history


class _Store:
    def __init__(self, runs, rows):
        self._runs = {r.id: r for r in runs}
        self._rows = rows

    def history(self, dataset, metric, catalog_id=None):
        return self._rows

    def get(self, run_id):
        return self._runs[run_id]


def _run(id, parent=None):
    return SimpleNamespace(id=id, parent_run_id=parent, short=id)


def test_a_delta_is_shown_only_against_a_parent_scored_on_the_same_or_later_version():
    runs = [_run("a"), _run("b", "a"), _run("c", "b"), _run("d")]
    store = _Store(runs, [("a", 1, 0.5), ("b", 2, 0.6), ("c", 1, 0.9), ("d", 3, 0.7)])
    rows = history.history(store, "cats", "val_accuracy")
    assert [(r.run.id, r.delta is not None, r.warm) for r in rows] == [
        ("a", False, False),  # a cold start compares to nothing
        ("b", True, True),  # continues a, on a later version: comparable
        ("c", False, True),  # continues b, but the version went backwards
        ("d", False, False),  # unchained
    ]
    assert rows[1].delta == pytest.approx(0.1)


def test_the_headline_metric_depends_on_the_task():
    assert history.headline_metric("span") == "val_span_f1"
    assert history.headline_metric("classification") == "val_accuracy"
    assert history.headline_metric("boxes") == "val_accuracy"
