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
from strata.contracts import AnyPrediction, feature_digest

from .active_learning import rank


@dataclass
class Coverage:
    """Each sample's declared features, and the samples that lack one.

    Keyed by checksum, because that is what a prediction is keyed by
    (``docs/adr/0006``): two samples with the same bytes are scored once,
    and are told the same features. See ``docs/adr/0052``.
    """

    by_checksum: dict[str, dict]
    #: Samples missing a declared feature, left out of the queue and said
    #: out loud. docs/adr/0011
    uncovered: list[SampleRow] = field(default_factory=list)
    #: Samples whose bytes an earlier sample in the pool holds with other
    #: features. One scoring pass tells a checksum one set, so these are
    #: left out and named rather than scored as the other. docs/adr/0052
    conflicting: list[SampleRow] = field(default_factory=list)

    @property
    def digests(self) -> dict[str, str]:
        """What keys a cached prediction: the third input, per sample. See ``docs/adr/0006``."""
        return {checksum: feature_digest(values) for checksum, values in self.by_checksum.items()}


def feature_values(catalog, pool: Sequence[SampleRow], specs) -> tuple[list[SampleRow], Coverage]:
    """Resolve the pool's features, dropping the samples a declaration does not cover.

    Decided per sample, since two samples can share bytes and not
    metadata (``docs/adr/0052``).
    """
    if not specs:
        return list(pool), Coverage({s.checksum: {} for s in pool})
    resolved = catalog.features_for([s.id for s in pool], specs)
    coverage = Coverage({})
    kept = []
    for s in pool:
        values = resolved.get(s.id, {})
        if not values:
            coverage.uncovered.append(s)
        elif coverage.by_checksum.setdefault(s.checksum, values) != values:
            coverage.conflicting.append(s)
        else:
            kept.append(s)
    return kept, coverage


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
    checksums = list(dict.fromkeys(s.checksum for s in pool))
    scores = cache.get(run_id, checksums, digests)
    # One prediction per checksum, whatever the type. docs/adr/0052
    missing = list({s.checksum: s for s in pool if s.checksum not in scores}.values())
    reused = len(scores)
    if missing:
        fresh = predict(paths_for(missing), [coverage.by_checksum[s.checksum] for s in missing])
        made = {s.checksum: p.value for s, p in zip(missing, fresh, strict=True)}
        cache.put(run_id, made, digests)
        scores.update(made)
    return Scored(scores, reused=reused, made=len(missing))


def disputed(catalog, label_set_id: int, collections, exclude: set[int]) -> list[SampleRow]:
    """Samples two people answered differently, as rows, minus ``exclude``.

    Found by sample id, which a dispute is recorded against: its checksum
    may be another type's bytes too. See ``docs/adr/0052``.
    """
    conflicts = catalog.conflicts.disputed(label_set_id, collections)
    wanted = {c["sample_id"] for c in conflicts} - exclude
    if not wanted:
        return []
    return [row for row in catalog.samples.matching(collections) if row.id in wanted]


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
