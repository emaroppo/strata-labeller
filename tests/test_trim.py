"""Repairing stored spans that start or end on whitespace.

What export now prevents, already in a catalog: a span selected in Label
Studio after aligned text, starting inside a run of spaces nobody could see.
"""

from strata.catalog import EVERYTHING, Catalog
from strata.contracts import Span, Spans, SpanSchema
from strata.labeller import trim

TEXT = "Location:   B2\nRoom: 4120\n"


def stored(tmp_path, *values: Spans, source: str = "human") -> tuple[Catalog, int, list[int]]:
    catalog = Catalog.local(tmp_path / "catalog")
    paths = []
    for i, _ in enumerate(values):
        path = tmp_path / f"doc{i}.txt"
        path.write_text(TEXT + f"#{i}")
        paths.append(path)
    ids = catalog.ingest(paths, media="text")
    label_set_id = catalog.label_sets.create("x", SpanSchema(classes=["LOC"]))
    for sample_id, value in zip(ids, values, strict=True):
        catalog.annotations.annotate(sample_id, label_set_id, value, source=source)
    return catalog, label_set_id, ids


def loc(start: int, end: int) -> Span:
    return Span(labels=["LOC"], start=start, end=end, text=TEXT[start:end])


def test_a_span_inside_a_run_of_spaces_is_planned_and_written(tmp_path):
    catalog, label_set_id, [sample_id] = stored(tmp_path, Spans(values=[loc(10, 14), loc(21, 25)]))

    report = trim.plan(catalog, label_set_id, EVERYTHING)
    assert (report.checked, len(report.changes), report.moved, report.dropped) == (1, 1, 1, 0)

    assert trim.apply(catalog, label_set_id, report) == 1
    assert catalog.annotations.annotation_of(sample_id, label_set_id) == Spans(
        values=[loc(12, 14), loc(21, 25)]
    )


def test_the_old_answer_stays_in_history_under_the_same_source(tmp_path):
    catalog, label_set_id, [sample_id] = stored(
        tmp_path, Spans(values=[loc(10, 14)]), source="import"
    )
    trim.apply(catalog, label_set_id, trim.plan(catalog, label_set_id, EVERYTHING))

    history = catalog.annotations.history(sample_id, label_set_id)
    assert [(a.source, a.current) for a in history] == [("import", False), ("import", True)]
    assert history[0].value == Spans(values=[loc(10, 14)])


def test_a_span_of_only_whitespace_is_dropped_and_counted(tmp_path):
    catalog, label_set_id, _ = stored(tmp_path, Spans(values=[loc(9, 12), loc(12, 14)]))
    report = trim.plan(catalog, label_set_id, EVERYTHING)
    assert (report.moved, report.dropped) == (0, 1)


def test_tight_spans_and_empty_answers_are_left_alone(tmp_path):
    catalog, label_set_id, _ = stored(tmp_path, Spans(values=[loc(12, 14)]), Spans())
    report = trim.plan(catalog, label_set_id, EVERYTHING)
    assert (report.checked, report.changes) == (1, [])


def test_the_document_is_read_with_its_carriage_returns():
    """Samples ingested before text was canonical keep \\r\\n, and offsets count
    it. Read as text, every offset after the first line would move.
    """
    from types import SimpleNamespace

    catalog = SimpleNamespace(blobs=SimpleNamespace(get=lambda location: b"Hi\r\n  Ana"))
    assert trim.text_of(catalog, SimpleNamespace(location=None)) == "Hi\r\n  Ana"
