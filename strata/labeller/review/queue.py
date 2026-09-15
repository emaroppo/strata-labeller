"""Building a review queue: what each sample is told, how it is scored, what goes first.

What ``push`` does between reading the pool and creating the tasks, with
nothing printed: each step returns what it decided and what it left out,
and the command says it. See ``docs/adr/0011`` for coverage and
``docs/adr/0012`` for the ordering.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from strata.catalog import SampleRow
from strata.labels import AnyPrediction, feature_digest

from .active_learning import rank


@dataclass
class Coverage:
    """Each sample's declared features, and the samples that lack one."""

    by_checksum: dict[str, dict]
    #: Samples missing a declared feature, left out of the queue and said
    #: out loud. docs/adr/0011
    uncovered: list[SampleRow] = field(default_factory=list)

    @property
    def digests(self) -> dict[str, str]:
        """What keys a cached prediction: the third input, per sample. See ``docs/adr/0006``."""
        return {checksum: feature_digest(values) for checksum, values in self.by_checksum.items()}


def feature_values(catalog, pool: Sequence[SampleRow], specs) -> tuple[list[SampleRow], Coverage]:
    """Resolve the pool's features, dropping the samples a declaration does not cover."""
    if not specs:
        return list(pool), Coverage({s.checksum: {} for s in pool})
    resolved = catalog.features_for([s.id for s in pool], specs)
    by_checksum = {s.checksum: resolved.get(s.id, {}) for s in pool}
    uncovered = [s for s in pool if not by_checksum[s.checksum]]
    dropped = {s.checksum for s in uncovered}
    kept = [s for s in pool if s.checksum not in dropped]
    return kept, Coverage({s.checksum: by_checksum[s.checksum] for s in kept}, uncovered)


@dataclass
class Scored:
    """Predictions for a pool, and where they came from."""

    scores: dict[str, AnyPrediction]
    reused: int = 0
    made: int = 0


def score_locally(
    store,
    cache,
    run_id: str,
    pool: Sequence[SampleRow],
    coverage: Coverage,
    paths_for: Callable[[list[SampleRow]], list[Path]],
    predict: Callable[[list[Path], list[dict]], list],
) -> Scored:
    """Score a pool with a local run, asking the cache first and filling it after.

    ``predict`` takes the paths and features of the samples the cache lacks
    and returns their predictions in order. Everything here deals in
    checksum -> prediction value: unwrapping the handler's result in some
    places but not others is how a cache came to hold values that read
    back empty.
    """
    digests = coverage.digests
    checksums = [s.checksum for s in pool]
    scores = cache.get(run_id, checksums, digests)
    missing = [s for s in pool if s.checksum not in scores]
    reused = len(scores)
    if missing:
        fresh = predict(paths_for(missing), [coverage.by_checksum[s.checksum] for s in missing])
        made = {s.checksum: p.value for s, p in zip(missing, fresh, strict=True)}
        cache.put(run_id, made, digests)
        scores.update(made)
    return Scored(scores, reused=reused, made=len(missing))


def disputed(catalog, label_set_id: int, collections, exclude: set[int]) -> list[SampleRow]:
    """Samples two people answered differently, as rows, minus ``exclude``."""
    conflicts = catalog.conflicts.disputed(label_set_id, collections)
    rows = (catalog.samples.by_checksum(c["checksum"]) for c in conflicts)
    return [row for row in rows if row is not None and row.id not in exclude]


@dataclass
class Queue:
    """What is sent, in order, with a score where there is one."""

    ranked: list[SampleRow]
    scored: dict[int, AnyPrediction]
    #: How many went first because they were answered two ways.
    disputed: int = 0


def plan(
    pool: Sequence[SampleRow],
    scores: dict[str, AnyPrediction],
    strategy: Callable,
    *,
    empty_share: float,
    disputed_rows: Sequence[SampleRow] = (),
    limit: int | None = None,
) -> Queue:
    """Order the queue: disputed samples first, then the pool by ``strategy``, cut at ``limit``.

    Disputed samples have an answer, so they are not in the pool; they are
    added, ahead of the ranking. See ``docs/adr/0009``.
    """
    if scores:
        ranked = rank(pool, scores, strategy, empty_share=empty_share)
        scored = {s.id: scores[s.checksum] for s in ranked}
    else:
        ranked, scored = list(pool), {}
    ranked = list(disputed_rows) + ranked
    if limit is not None:
        ranked = ranked[:limit]
    return Queue(ranked, scored, disputed=len(disputed_rows))
