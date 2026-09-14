"""Reading a project's training history: the rows a report shows, and the rule behind its deltas."""

from dataclasses import dataclass, field

from strata.modelling import Run, Unchecked

#: The number that summarises a run, by task. Defaulting to the
#: classification one meant a span project reported nothing at all: every
#: run had metrics, just not that name.
HEADLINE_METRIC = {
    "classification": "val_accuracy",
    "span": "val_span_f1",
}


def headline_metric(task: str) -> str:
    return HEADLINE_METRIC.get(task, "val_accuracy")


@dataclass
class HistoryRow:
    run: Run
    value: float
    version: int | None
    #: Against the run's own parent, and only when both were scored on the
    #: same held-out samples. None means "cannot be compared", which is not
    #: the same statement as "did not move".
    delta: float | None
    #: Whether it continued a run in the store. False is a cold start, or a
    #: round imported from before the store existed.
    warm: bool
    #: How much of what the run saw nobody checked, per side and batch.
    #: Empty for a run recorded before that was written down.
    unchecked: list[Unchecked] = field(default_factory=list)


def history(store, dataset: str, metric: str, catalog_id: str | None = None) -> list[HistoryRow]:
    """One row per run that recorded ``metric`` over ``dataset``, oldest first.

    Within ``catalog_id`` when given: a project's run store can hold runs
    from a catalog it no longer names, and a delta across the two
    measures nothing.

    The delta rule is the whole reason this is not a plain dump of the
    store: a warm-started number means something against its parent and
    nothing against a run from another lineage. A dataset version going
    backwards means the lineage crossed in from the old layout, where the
    split was recomputed every round, and comparing those measured nothing
    but a change of validation set.
    """
    rows: list[HistoryRow] = []
    seen: dict[str, float] = {}
    versions: dict[str, int | None] = {}
    for run_id, version, value in store.history(dataset, metric, catalog_id):
        run = store.get(run_id)
        parent = run.parent_run_id
        before, now = versions.get(parent), version
        comparable = parent in seen and before is not None and now is not None and before <= now
        rows.append(
            HistoryRow(
                run=run,
                value=value,
                version=version,
                delta=(value - seen[parent]) if comparable else None,
                warm=bool(parent),
                unchecked=store.unchecked(run_id),
            )
        )
        seen[run_id] = value
        versions[run_id] = version
    return rows


def unchecked_share(rows: list[Unchecked], side: str) -> tuple[int, int] | None:
    """``(unreviewed, samples)`` on one side, or None when the run did not say.

    Validation is the side that matters: a validation set of unreviewed
    imports measures agreement with whoever labelled them, not accuracy.
    """
    on_side = [r for r in rows if r.side == side]
    if not on_side:
        return None
    return sum(r.unreviewed for r in on_side), sum(r.samples for r in on_side)


def unchecked_json(rows: list[Unchecked]) -> list[dict]:
    return [
        {"side": r.side, "batch": r.batch, "samples": r.samples, "unreviewed": r.unreviewed}
        for r in rows
    ]


def run_json(run) -> dict:
    """One run, as something to compute on rather than to read.

    Everything the store holds, including ``params`` and ``classes``. Those
    are the fields that answer whether two runs are asking the same
    question: a metric moved by changing the data, the model, or what the
    model was told to do, and only the last of those is invisible in a
    table of numbers.
    """
    data = run.model_dump(mode="json")
    # Not a field on the model, and the one thing a caller would otherwise
    # have to reimplement the id format to get.
    data["short"] = run.short
    return data


def run_detail(store, run) -> dict:
    """A run, its chain, and its curve. The curve only here: epochs times metrics."""
    return {
        "run": run_json(run),
        "chain": [run_json(r) for r in store.chain(run.id)],
        "curve": [{"epoch": epoch, **reported} for epoch, reported in store.curve(run.id)],
        "unchecked": unchecked_json(store.unchecked(run.id)),
    }


def history_json(dataset: str, metric: str, rows: list[HistoryRow]) -> dict:
    """The history as a script reads it: every field, deltas null where incomparable."""
    return {
        "dataset": dataset,
        "metric": metric,
        "runs": [
            {
                **run_json(row.run),
                "value": row.value,
                "delta": row.delta,
                "lineage": "warm" if row.warm else "unchained",
                "unchecked": unchecked_json(row.unchecked),
            }
            for row in rows
        ],
    }
