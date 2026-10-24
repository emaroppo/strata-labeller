"""Keeping Label Studio and the catalog in step.

No Label Studio here: what is under test is the translation either way, so
these work on the payloads the SDK hands over and hands back.
"""

import pytest

from strata.catalog import EVERYTHING, Catalog
from strata.contracts import Choices, ClassificationSchema
from strata.labeller.labelstudio.adapter import TYPE_KEY, Addressing, blob_url
from strata.labeller.labelstudio.sync import (
    load_task_map,
    pull_annotations,
    rebuild_task_map,
    save_task_map,
    tasks_to_push,
)

ADDRESSING = Addressing(prefix="blobs")


@pytest.fixture
def schema(project):
    return project.schema


@pytest.fixture
def stocked(catalog, files):
    paths = files(5)
    ids = catalog.ingest(paths, media="image")
    label_set_id = catalog.label_sets.create(
        "presence", ClassificationSchema(classes=["cat", "dog"])
    )
    return catalog, ids, label_set_id


def ls_task(task_id: int, url: str, data_key: str = "image", **extra) -> dict:
    return {"id": task_id, "data": {data_key: url}, **extra}


def typed(task: dict, type_: str) -> dict:
    return {**task, "data": {**task["data"], TYPE_KEY: type_}}


@pytest.fixture
def held_twice(tmp_path):
    """Give some checksums' bytes a second sample, of another type.

    The same bytes as two types are two samples, so a lookup by checksum
    alone no longer has one answer. See ``docs/adr/0052``.
    """

    def _hold(catalog, *checksums):
        cached = catalog.ensure_cached(list(checksums), tmp_path / "held")
        catalog.ingest([cached[c] for c in checksums], media="image", subtype="copy")

    return _hold


# ----------------------------------------------------------------------
# The task map
# ----------------------------------------------------------------------


def test_a_task_map_round_trips(project):
    save_task_map(project, 7, {1: 100, 2: 200})
    # JSON keys are strings and sample ids are not, which is a quiet way to
    # lose a whole map
    assert load_task_map(project, 7) == {1: 100, 2: 200}


def test_a_missing_task_map_is_empty(project):
    assert load_task_map(project, 7) == {}


def test_task_maps_are_kept_per_label_studio_project(project):
    save_task_map(project, 1, {1: 100})
    save_task_map(project, 2, {1: 999})
    assert load_task_map(project, 1) == {1: 100}


def test_the_map_rebuilds_from_what_label_studio_holds(stocked, schema):
    catalog, _ids, label_set_id = stocked
    samples = catalog.samples.unlabelled(label_set_id, EVERYTHING)
    tasks = [ls_task(500 + i, blob_url(s, "blobs")) for i, s in enumerate(samples)]

    mapping, unrecognised, _ = rebuild_task_map(tasks, catalog, ADDRESSING, schema.data_key)
    # The cache only saves a listing; nothing depends on it surviving
    assert mapping == {s.id: 500 + i for i, s in enumerate(samples)}
    assert unrecognised == []


def test_a_task_from_before_the_cutover_is_reported_not_guessed(stocked, schema):
    catalog, _, _ = stocked
    tasks = [ls_task(1, "/data/local-files/?d=images/vid1/f001.jpg")]

    mapping, unrecognised, _ = rebuild_task_map(tasks, catalog, ADDRESSING, schema.data_key)
    # It points at the old data root, which names no blob. Matching it to
    # some sample by string similarity would be worse than saying so.
    assert mapping == {}
    assert len(unrecognised) == 1


def test_a_blob_no_longer_in_the_catalog_is_unrecognised(stocked, schema):
    catalog, _, _ = stocked
    tasks = [ls_task(1, "/data/local-files/?d=blobs/aa/bb/" + "0" * 64 + ".jpg")]
    mapping, unrecognised, _ = rebuild_task_map(tasks, catalog, ADDRESSING, schema.data_key)
    assert mapping == {} and len(unrecognised) == 1


def test_a_typed_task_finds_the_sample_of_its_type(stocked, schema):
    catalog, _, label_set_id = stocked
    [sample] = catalog.samples.unlabelled(label_set_id, EVERYTHING)[:1]
    tasks = [typed(ls_task(1, blob_url(sample, "blobs")), sample.type)]

    mapping, unrecognised, ambiguous = rebuild_task_map(tasks, catalog, ADDRESSING, schema.data_key)
    assert mapping == {sample.id: 1}
    assert unrecognised == [] and ambiguous == []


def test_a_typed_task_never_settles_for_another_type(stocked, schema):
    catalog, _, label_set_id = stocked
    [sample] = catalog.samples.unlabelled(label_set_id, EVERYTHING)[:1]
    tasks = [typed(ls_task(1, blob_url(sample, "blobs")), "text/generated")]

    mapping, unrecognised, _ = rebuild_task_map(tasks, catalog, ADDRESSING, schema.data_key)
    # The bytes are catalogued, but not as that type: the one row there is
    # not what the task showed
    assert mapping == {}
    assert len(unrecognised) == 1


def test_a_task_from_before_types_resolves_while_one_sample_holds_its_bytes(stocked, schema):
    catalog, _, label_set_id = stocked
    [sample] = catalog.samples.unlabelled(label_set_id, EVERYTHING)[:1]
    tasks = [ls_task(1, blob_url(sample, "blobs"))]

    mapping, _, ambiguous = rebuild_task_map(tasks, catalog, ADDRESSING, schema.data_key)
    assert mapping == {sample.id: 1}
    assert ambiguous == []


def test_a_task_from_before_types_is_refused_by_id_once_two_samples_hold_its_bytes(
    stocked, schema, held_twice
):
    catalog, _, label_set_id = stocked
    first, second = catalog.samples.unlabelled(label_set_id, EVERYTHING)[:2]
    held_twice(catalog, first.checksum)
    tasks = [
        ls_task(7, blob_url(first, "blobs")),
        typed(ls_task(8, blob_url(first, "blobs")), first.type),
        ls_task(9, blob_url(second, "blobs")),
    ]

    mapping, unrecognised, ambiguous = rebuild_task_map(tasks, catalog, ADDRESSING, schema.data_key)
    # Matching it to whichever row came first would be a guess; the rest of
    # the listing still places
    assert ambiguous == [7]
    assert mapping == {first.id: 8, second.id: 9}
    assert unrecognised == []


# ----------------------------------------------------------------------
# Pushing
# ----------------------------------------------------------------------


def test_everything_unseen_is_pushed(stocked, schema):
    catalog, _ids, label_set_id = stocked
    samples = catalog.samples.unlabelled(label_set_id, EVERYTHING)
    tasks, report = tasks_to_push(samples, catalog, label_set_id, schema, ADDRESSING, {})
    assert report.pushed == 5
    assert len(tasks) == 5


def test_samples_label_studio_already_has_are_skipped(stocked, schema):
    catalog, _ids, label_set_id = stocked
    samples = catalog.samples.unlabelled(label_set_id, EVERYTHING)
    existing = {samples[0].id: 100, samples[1].id: 101}

    tasks, report = tasks_to_push(samples, catalog, label_set_id, schema, ADDRESSING, existing)
    # What makes a push resumable: an interrupted one is just run again
    assert report.pushed == 3
    assert report.already_present == 2
    assert all(t.sample_id not in existing for t in tasks)


def test_a_pushed_task_points_at_the_blob(stocked, schema):
    catalog, _ids, label_set_id = stocked
    samples = catalog.samples.unlabelled(label_set_id, EVERYTHING)
    tasks, _ = tasks_to_push(samples[:1], catalog, label_set_id, schema, ADDRESSING, {})
    assert samples[0].checksum in tasks[0].data[schema.data_key]


def test_a_pushed_task_names_its_samples_type(stocked, schema):
    catalog, _ids, label_set_id = stocked
    samples = catalog.samples.unlabelled(label_set_id, EVERYTHING)
    tasks, _ = tasks_to_push(samples, catalog, label_set_id, schema, ADDRESSING, {})
    # Always, not only where two samples share the bytes. docs/adr/0052
    assert [t.data[TYPE_KEY] for t in tasks] == [s.type for s in samples]
    assert all(t.data[TYPE_KEY] == "image/plain" for t in tasks)


# ----------------------------------------------------------------------
# Pulling
# ----------------------------------------------------------------------


def annotated(task_id, url, choices, **extra):
    return ls_task(
        task_id,
        url,
        annotations=[
            {
                "was_cancelled": False,
                "result": [
                    {
                        "from_name": "label",
                        "to_name": "image",
                        "type": "choices",
                        "value": {"choices": choices},
                    }
                ]
                if choices is not None
                else [],
                **extra,
            }
        ],
    )


def test_an_annotation_comes_back_as_a_value(stocked, schema):
    catalog, _ids, label_set_id = stocked
    [sample] = catalog.samples.unlabelled(label_set_id, EVERYTHING)[:1]
    exported = [annotated(1, blob_url(sample, "blobs"), ["cat"])]

    items, report = pull_annotations(
        exported, catalog, label_set_id, schema, ADDRESSING, ["cat", "dog"]
    )
    assert items == [(sample.id, Choices(values=["cat"]))]
    assert report.annotated == 1


def test_an_empty_annotation_comes_back_as_an_answer(stocked, schema):
    catalog, _ids, label_set_id = stocked
    [sample] = catalog.samples.unlabelled(label_set_id, EVERYTHING)[:1]
    exported = [annotated(1, blob_url(sample, "blobs"), [])]

    items, report = pull_annotations(
        exported, catalog, label_set_id, schema, ADDRESSING, ["cat", "dog"]
    )
    # A reviewer who found none of the classes present has answered
    assert items == [(sample.id, Choices())]
    assert report.annotated == 1


def test_a_task_nobody_has_answered_is_left_alone(stocked, schema):
    catalog, _ids, label_set_id = stocked
    [sample] = catalog.samples.unlabelled(label_set_id, EVERYTHING)[:1]
    exported = [ls_task(1, blob_url(sample, "blobs"))]

    items, report = pull_annotations(exported, catalog, label_set_id, schema, ADDRESSING, ["cat"])
    # Writing an empty value here would claim someone had looked
    assert items == []
    assert report.total == 0


def test_a_cancelled_annotation_is_a_skip(stocked, schema):
    catalog, _ids, label_set_id = stocked
    [sample] = catalog.samples.unlabelled(label_set_id, EVERYTHING)[:1]
    exported = [
        ls_task(
            1,
            blob_url(sample, "blobs"),
            annotations=[{"was_cancelled": True, "result": []}],
        )
    ]

    items, report = pull_annotations(exported, catalog, label_set_id, schema, ADDRESSING, ["cat"])
    assert items == [(sample.id, None)]
    assert report.skipped == 1


def test_a_class_nobody_declared_is_reported(stocked, schema):
    catalog, _ids, label_set_id = stocked
    [sample] = catalog.samples.unlabelled(label_set_id, EVERYTHING)[:1]
    exported = [annotated(1, blob_url(sample, "blobs"), ["cat", "fox"])]

    _, report = pull_annotations(
        exported, catalog, label_set_id, schema, ADDRESSING, ["cat", "dog"]
    )
    # Someone added a class in the Label Studio UI; the catalog will refuse
    # it, so it has to be visible rather than an error mid-write
    assert report.undeclared == {"fox"}


def test_volatile_fields_do_not_survive(stocked, schema):
    catalog, _ids, label_set_id = stocked
    [sample] = catalog.samples.unlabelled(label_set_id, EVERYTHING)[:1]
    exported = [
        ls_task(
            1,
            blob_url(sample, "blobs"),
            annotations=[
                {
                    "was_cancelled": False,
                    "result": [
                        {
                            "id": "abc123",
                            "lead_time": 4.2,
                            "from_name": "label",
                            "to_name": "image",
                            "type": "choices",
                            "value": {"choices": ["dog"]},
                        }
                    ],
                }
            ],
        )
    ]
    items, _ = pull_annotations(exported, catalog, label_set_id, schema, ADDRESSING, ["cat", "dog"])
    # They say nothing about the annotation and would churn the store
    assert items == [(sample.id, Choices(values=["dog"]))]


def test_an_unrecognised_task_is_reported_not_dropped(stocked, schema):
    catalog, _, label_set_id = stocked
    exported = [annotated(1, "/data/local-files/?d=images/old.jpg", ["cat"])]
    items, report = pull_annotations(exported, catalog, label_set_id, schema, ADDRESSING, ["cat"])
    assert items == []
    assert len(report.unrecognised) == 1


def test_a_pushed_task_comes_back_to_its_own_sample(stocked, schema):
    catalog, _ids, label_set_id = stocked
    [sample] = catalog.samples.unlabelled(label_set_id, EVERYTHING)[:1]
    [task], _ = tasks_to_push([sample], catalog, label_set_id, schema, ADDRESSING, {})
    exported = [{**annotated(1, "", ["cat"]), "data": task.data}]

    items, report = pull_annotations(exported, catalog, label_set_id, schema, ADDRESSING, ["cat"])
    assert items == [(sample.id, Choices(values=["cat"]))]
    assert report.ambiguous == []


def test_an_answer_on_a_task_from_before_types_is_refused_by_id_when_ambiguous(
    stocked, schema, held_twice
):
    catalog, _ids, label_set_id = stocked
    first, second = catalog.samples.unlabelled(label_set_id, EVERYTHING)[:2]
    held_twice(catalog, first.checksum)
    exported = [
        annotated(3, blob_url(first, "blobs"), ["cat"]),
        annotated(4, blob_url(second, "blobs"), ["dog"]),
    ]

    items, report = pull_annotations(
        exported, catalog, label_set_id, schema, ADDRESSING, ["cat", "dog"]
    )
    # Not written to either sample, not dropped without a word, and the
    # export carries on with the tasks it can place
    assert report.ambiguous == [3]
    assert report.unrecognised == []
    assert items == [(second.id, Choices(values=["dog"]))]


def test_reviewed_only_keeps_back_an_answer_nobody_opened(stocked, schema):
    """A seeded project hands its own guesses back as ground truth.

    Tasks imported with annotations arrive already answered. Export writes
    everything it finds as a human answer, so exporting part-way through a
    review stamps the seed on every task still untouched — and a seed can
    be very wrong: one measured here found none of the people and none of
    the places a reviewer went on to mark.
    """
    catalog, _ids, label_set_id = stocked
    [sample] = catalog.samples.unlabelled(label_set_id, EVERYTHING)[:1]
    exported = [annotated(1, blob_url(sample, "blobs"), ["cat"])]

    items, report = pull_annotations(
        exported, catalog, label_set_id, schema, ADDRESSING, ["cat"], reviewed_only=True
    )
    assert items == []
    assert report.untouched == 1
    assert report.annotated == 0


def test_reviewed_only_takes_an_answer_somebody_worked_on(stocked, schema):
    catalog, _ids, label_set_id = stocked
    [sample] = catalog.samples.unlabelled(label_set_id, EVERYTHING)[:1]
    exported = [annotated(1, blob_url(sample, "blobs"), ["cat"], lead_time=42.0)]

    items, report = pull_annotations(
        exported, catalog, label_set_id, schema, ADDRESSING, ["cat"], reviewed_only=True
    )
    assert items == [(sample.id, Choices(values=["cat"]))]
    assert report.untouched == 0


def test_without_the_flag_everything_still_comes_back(stocked, schema):
    # The default is unchanged: a project nobody seeded has no such problem
    catalog, _ids, label_set_id = stocked
    [sample] = catalog.samples.unlabelled(label_set_id, EVERYTHING)[:1]
    exported = [annotated(1, blob_url(sample, "blobs"), ["cat"])]

    items, _report = pull_annotations(exported, catalog, label_set_id, schema, ADDRESSING, ["cat"])
    assert items == [(sample.id, Choices(values=["cat"]))]


def test_an_undeclared_class_is_found_whatever_kind_of_value_carries_it(tmp_path):
    """Reading a value's `values` as class names is a classification-ism.

    Spans carry their class on each labelled range, so the old check raised
    a TypeError comparing Span objects instead of naming the class nobody
    declared — and it raised on every span export, declared or not.
    """
    from strata.labeller.labelstudio.schemas.span import SpanSchema as LSSpan

    catalog = Catalog.local(tmp_path / "catalog")
    source = tmp_path / "doc.txt"
    source.write_text("Ada Lovelace wrote it")
    [_sample_id] = catalog.ingest([source], media="text")
    schema = LSSpan(classes=["PER"])
    label_set_id = catalog.label_sets.create("x", schema.catalog_schema())
    [row] = catalog.samples.unlabelled(label_set_id, EVERYTHING)

    task = ls_task(
        1,
        blob_url(row, "blobs"),
        data_key="text",
        annotations=[
            {
                "was_cancelled": False,
                "lead_time": 12.0,
                "result": [
                    {
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                            "start": 0,
                            "end": 12,
                            "text": "Ada Lovelace",
                            "labels": ["NOBODY_DECLARED_THIS"],
                        },
                    }
                ],
            }
        ],
    )
    _items, report = pull_annotations([task], catalog, label_set_id, schema, ADDRESSING, ["PER"])
    assert report.undeclared == {"NOBODY_DECLARED_THIS"}


def test_a_span_starting_inside_a_run_of_spaces_comes_in_trimmed(tmp_path):
    """Label Studio shows a run of spaces as one, so a selection made after
    aligned text starts inside the run. The reviewer selected ``B2``.
    """
    from strata.contracts import Span, Spans
    from strata.labeller.labelstudio.schemas.span import SpanSchema as LSSpan

    catalog = Catalog.local(tmp_path / "catalog")
    source = tmp_path / "doc.txt"
    source.write_text("Location:   B2\nRoom: 4120")
    catalog.ingest([source], media="text")
    schema = LSSpan(classes=["LOC"])
    label_set_id = catalog.label_sets.create("x", schema.catalog_schema())
    [row] = catalog.samples.unlabelled(label_set_id, EVERYTHING)

    def result(start, end, text):
        return {
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {"start": start, "end": end, "text": text, "labels": ["LOC"]},
        }

    task = ls_task(
        1,
        blob_url(row, "blobs"),
        data_key="text",
        annotations=[
            {
                "was_cancelled": False,
                "lead_time": 9.0,
                "result": [result(10, 14, "  B2"), result(21, 25, "4120")],
            }
        ],
    )
    items, report = pull_annotations([task], catalog, label_set_id, schema, ADDRESSING, ["LOC"])
    assert report.trimmed == 1
    assert items == [
        (
            row.id,
            Spans(
                values=[
                    Span(labels=["LOC"], start=12, end=14, text="B2"),
                    Span(labels=["LOC"], start=21, end=25, text="4120"),
                ]
            ),
        )
    ]
