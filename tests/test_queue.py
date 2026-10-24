"""The queue as a plan: what goes first, what is left out, what a cache is asked."""

from pathlib import Path

from strata.catalog import Location, SampleRow
from strata.contracts import ChoicesPrediction
from strata.labeller.review import queue
from strata.labeller.review.active_learning import least_confident


def _row(i: int) -> SampleRow:
    return SampleRow(i, f"sum{i}", Location(f"f{i}", 0, 1), "image", "plain", None)


def _twin(i: int, of: int) -> SampleRow:
    """Another sample over ``of``'s bytes, as another type. docs/adr/0052"""
    return SampleRow(i, f"sum{of}", Location(f"f{of}", 0, 1), "text", "generated", None)


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


def test_a_sample_sharing_bytes_with_an_uncovered_one_is_still_kept():
    pool = [_row(1), _twin(2, of=1)]
    kept, coverage = queue.feature_values(_Catalog({2: {"species": ["a"]}}), pool, specs=[object()])
    # Coverage is each sample's own: the one lacking the feature does not
    # take its twin out of the queue with it
    assert [s.id for s in kept] == [2]
    assert [s.id for s in coverage.uncovered] == [1]
    assert coverage.by_checksum == {"sum1": {"species": ["a"]}}


def test_a_sample_told_other_features_than_its_twin_is_left_out_and_named():
    pool = [_row(1), _twin(2, of=1)]
    resolved = {1: {"species": ["a"]}, 2: {"species": ["b"]}}
    kept, coverage = queue.feature_values(_Catalog(resolved), pool, specs=[object()])
    # A pass tells a checksum one set of features; scoring the second as
    # the first would be quietly wrong
    assert [s.id for s in kept] == [1]
    assert [s.id for s in coverage.conflicting] == [2]
    assert coverage.by_checksum == {"sum1": {"species": ["a"]}}


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


def test_two_samples_with_the_same_bytes_are_predicted_once():
    pool = [_row(1), _twin(2, of=1)]
    cache = _Cache({})
    _, coverage = queue.feature_values(_Catalog({}), pool, specs=[])

    class Scored:
        def __init__(self, value):
            self.value = value

    calls = []

    def predict(paths, features):
        calls.append(paths)
        return [Scored(_sure(0.3)) for _ in paths]

    scored = queue.score_locally(
        None,
        cache,
        "run",
        pool,
        coverage,
        paths_for=lambda rows: [Path(r.checksum) for r in rows],
        predict=predict,
    )
    # A prediction is the bytes', whatever the type. docs/adr/0052
    assert calls == [[Path("sum1")]]
    assert scored.made == 1 and set(scored.scores) == {"sum1"}

    planned = queue.plan(pool, scored.scores, least_confident, empty_share=1.0)
    assert set(planned.scored) == {1, 2}


class _Disputes:
    def __init__(self, rows, disputed_ids):
        self.rows = rows
        self.disputed_ids = disputed_ids
        self.conflicts = self
        self.samples = self

    def disputed(self, label_set_id, collections):
        return [
            {"sample_id": r.id, "checksum": r.checksum}
            for r in self.rows
            if r.id in self.disputed_ids
        ]

    def matching(self, collections):
        return self.rows

    def by_checksum(self, checksum, type=None):
        raise AssertionError("a dispute names its sample by id")


def test_a_disputed_sample_is_found_by_its_id_not_its_bytes():
    rows = [_row(1), _twin(2, of=1), _row(3)]
    found = queue.disputed(_Disputes(rows, {2}), 1, ["demo"], exclude=set())
    # Its bytes are another sample's too; the dispute is this one's
    assert [r.id for r in found] == [2]


def test_nothing_disputed_reads_no_rows():
    class _NoRows(_Disputes):
        def matching(self, collections):
            raise AssertionError("nothing to look up")

    assert queue.disputed(_NoRows([_row(1)], {1}), 1, ["demo"], exclude={1}) == []
