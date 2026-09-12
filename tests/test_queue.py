"""The queue as a plan: what goes first, what is left out, what a cache is asked."""

from strata.catalog import Location, SampleRow
from strata.labeller import queue
from strata.labeller.active_learning import least_confident
from strata.labels import ChoicesPrediction


def _row(i: int) -> SampleRow:
    return SampleRow(i, f"sum{i}", Location(f"f{i}", 0, 1), "image", "plain", None)


def _sure(p: float) -> ChoicesPrediction:
    return ChoicesPrediction(values=["cat"], confidences=[p])


def test_disputed_samples_go_first_and_a_limit_keeps_them():
    pool = [_row(1), _row(2), _row(3)]
    scores = {"sum1": _sure(0.9), "sum2": _sure(0.2), "sum3": _sure(0.5)}
    planned = queue.plan(
        pool, scores, least_confident, empty_share=0.0, disputed_rows=[_row(9)], limit=2
    )
    assert [s.id for s in planned.ranked] == [9, 2]
    assert planned.disputed == 1
    # A disputed sample carries no score; the ranked ones do
    assert set(planned.scored) == {1, 2, 3}


def test_without_scores_the_pool_is_sent_as_it_came():
    pool = [_row(1), _row(2)]
    planned = queue.plan(pool, {}, least_confident, empty_share=0.0)
    assert [s.id for s in planned.ranked] == [1, 2]
    assert planned.scored == {}


class _Catalog:
    def __init__(self, resolved):
        self._resolved = resolved

    def features_for(self, ids, specs):
        return {i: self._resolved[i] for i in ids if i in self._resolved}


def test_a_sample_missing_a_declared_feature_is_left_out_and_named():
    pool = [_row(1), _row(2)]
    kept, coverage = queue.feature_values(_Catalog({1: {"species": ["a"]}}), pool, specs=[object()])
    assert [s.id for s in kept] == [1]
    assert [s.id for s in coverage.uncovered] == [2]
    assert coverage.by_checksum == {"sum1": {"species": ["a"]}}
    assert set(coverage.digests) == {"sum1"}


def test_no_declared_features_means_every_sample_is_covered_with_nothing():
    pool = [_row(1)]
    kept, coverage = queue.feature_values(_Catalog({}), pool, specs=[])
    assert kept == pool and coverage.uncovered == [] and coverage.by_checksum == {"sum1": {}}


class _Cache:
    def __init__(self, held):
        self.held = held
        self.put_calls = []

    def get(self, run_id, checksums, digests):
        return {c: self.held[c] for c in checksums if c in self.held}

    def put(self, run_id, made, digests):
        self.put_calls.append(made)


def test_scoring_locally_asks_the_cache_first_and_fills_it_after():
    pool = [_row(1), _row(2)]
    cache = _Cache({"sum1": _sure(0.7)})
    _, coverage = queue.feature_values(_Catalog({}), pool, specs=[])

    class Scored:
        def __init__(self, value):
            self.value = value

    calls = []

    def predict(paths, features):
        calls.append((paths, features))
        return [Scored(_sure(0.3)) for _ in paths]

    scored = queue.score_locally(
        None,
        cache,
        "run",
        pool,
        coverage,
        paths_for=lambda rows: [r.checksum for r in rows],
        predict=predict,
    )
    assert scored.reused == 1 and scored.made == 1
    assert set(scored.scores) == {"sum1", "sum2"}
    assert calls == [(["sum2"], [{}])]
    assert cache.put_calls == [{"sum2": scored.scores["sum2"]}]
