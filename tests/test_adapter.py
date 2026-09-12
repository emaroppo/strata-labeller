"""The Label Studio boundary.

Everything Label Studio shaped has to stop here, so these tests are mostly
about what crosses and in what form — and about the two cases that are easy
to conflate, an empty answer and no answer.
"""

import pytest

from strata.catalog import EVERYTHING
from strata.labeller.adapter import (
    LOCAL_FILES,
    Addressing,
    blob_url,
    build_tasks,
    checksum_from_url,
    from_results,
    to_results,
)
from strata.labels import Choices, ChoicesPrediction

ADDRESSING = Addressing(prefix="blobs")


@pytest.fixture
def schema(project):
    return project.schema


@pytest.fixture
def stocked(catalog, files):
    """A catalog with samples and a label set over them."""
    from strata.labels import ClassificationSchema

    paths = files(4)
    ids = catalog.ingest(paths, media="image")
    label_set_id = catalog.label_sets.create(
        "presence", ClassificationSchema(classes=["cat", "dog"])
    )
    return ids, label_set_id


# ----------------------------------------------------------------------
# Values
# ----------------------------------------------------------------------


def test_a_value_becomes_label_studio_results(schema):
    results = to_results(Choices(values=["cat", "dog"]), schema)
    assert results[0]["type"] == "choices"
    assert results[0]["value"]["choices"] == ["cat", "dog"]


def test_results_become_a_value(schema):
    results = to_results(Choices(values=["cat"]), schema)
    assert from_results(results, schema) == Choices(values=["cat"])


def test_a_value_survives_a_round_trip(schema):
    value = Choices(values=["cat", "dog"])
    assert from_results(to_results(value, schema), schema) == value


def test_an_empty_value_produces_no_results(schema):
    # Label Studio has no way to say "nothing", and does not need one: an
    # annotation with an empty result list is exactly that
    assert to_results(Choices(), schema) == []


def test_no_results_is_an_empty_value_rather_than_nothing(schema):
    # The distinction the catalog keeps: a reviewer who found none of the
    # classes present has answered, and that is not the same as never having
    # been asked. Absence of a row is the second; this is the first.
    assert from_results([], schema) == Choices()


def test_the_control_names_come_from_the_schema(schema):
    [result] = to_results(Choices(values=["cat"]), schema)
    assert result["from_name"] == schema.from_name
    assert result["to_name"] == schema.to_name


def test_a_prediction_crosses_as_its_values(schema):
    from strata.labeller.adapter import prediction_to_results

    prediction = ChoicesPrediction(values=["cat"], confidences=[0.9])
    assert prediction_to_results(prediction, schema)[0]["value"]["choices"] == ["cat"]


# ----------------------------------------------------------------------
# Addressing
# ----------------------------------------------------------------------


def test_a_url_addresses_the_blob(catalog, stocked, tmp_path):
    _, label_set_id = stocked
    [sample] = catalog.unlabelled(label_set_id, EVERYTHING)[:1]
    url = blob_url(sample, "blobs")
    # The checksum is in the path, so the URL names one sample rather than
    # matching a string that might mean several things
    assert sample.checksum in url
    assert url.startswith(LOCAL_FILES)


def test_a_url_round_trips_to_its_sample(catalog, stocked):
    _, label_set_id = stocked
    [sample] = catalog.unlabelled(label_set_id, EVERYTHING)[:1]
    url = blob_url(sample, "blobs")
    assert checksum_from_url(url, "blobs") == sample.checksum


def test_a_url_survives_the_bytes_moving(catalog, stocked):
    _, label_set_id = stocked
    [sample] = catalog.unlabelled(label_set_id, EVERYTHING)[:1]
    before = blob_url(sample, "blobs")

    # What repacking into shards does to a row. Every task in Label Studio
    # was created before this happened, so a URL that stopped resolving here
    # would strand every annotation in progress.
    from dataclasses import replace

    from strata.catalog import Location

    moved = replace(sample, location=Location("shards/abc123.tar", 91136, 4096))
    assert blob_url(moved, "blobs") == before
    assert checksum_from_url(before, "blobs") == sample.checksum


def test_a_url_is_percent_encoded():
    from strata.catalog import Location

    class Odd:
        checksum = "ab" * 32
        location = Location("shards/x.tar", 0, 1)
        metadata = {"source_path": "/raw/file name&x.jpg"}

    url = blob_url(Odd(), "blobs")
    assert " " not in url and "&" not in url.split("?d=", 1)[1]
    assert checksum_from_url(url, "blobs") == "ab" * 32


def test_a_url_from_before_the_catalog_names_no_blob():
    # A task created against the old data root: it points somewhere real,
    # just not at a blob, and that has to be distinguishable
    old = f"{LOCAL_FILES}images/vid1/f001.jpg"
    assert checksum_from_url(old, "blobs") is None


def test_something_that_is_not_a_local_file_names_no_blob():
    assert checksum_from_url("https://example.com/photo.jpg", "blobs") is None


def test_a_path_under_the_prefix_that_is_not_a_digest_names_no_blob():
    # Reading it as a checksum would send it to the catalog as a lookup that
    # cannot match, reporting "unrecognised" for what is really a bad prefix
    assert checksum_from_url(f"{LOCAL_FILES}blobs/notes.txt", "blobs") is None


# ----------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------


def test_a_task_carries_the_sample_it_came_from(catalog, stocked, schema):
    ids, label_set_id = stocked
    samples = catalog.unlabelled(label_set_id, EVERYTHING)
    tasks = build_tasks(samples, catalog, label_set_id, schema, ADDRESSING)
    assert {t.sample_id for t in tasks} == set(ids)


def test_a_task_points_at_the_right_key_for_the_media(catalog, stocked, schema):
    _, label_set_id = stocked
    [task] = build_tasks(
        catalog.unlabelled(label_set_id, EVERYTHING)[:1], catalog, label_set_id, schema, ADDRESSING
    )
    assert schema.data_key in task.data


def test_an_unannotated_task_carries_no_annotation(catalog, stocked, schema):
    _, label_set_id = stocked
    [task] = build_tasks(
        catalog.unlabelled(label_set_id, EVERYTHING)[:1], catalog, label_set_id, schema, ADDRESSING
    )
    assert task.annotations == []
    assert not task.answered
    assert "annotations" not in task.as_import()


def test_an_annotated_task_arrives_answered(catalog, stocked, schema):
    ids, label_set_id = stocked
    catalog.annotations.annotate(ids[0], label_set_id, Choices(values=["cat"]))
    samples = [s for s in catalog.labelled(label_set_id, EVERYTHING)]
    [task] = build_tasks(samples, catalog, label_set_id, schema, ADDRESSING)

    # Label Studio is a view of the catalog rather than a second copy, so
    # rebuilding a project must not ask again for what is already answered
    assert task.as_import()["annotations"] == [{"result": task.annotations}]
    assert task.annotations[0]["value"]["choices"] == ["cat"]


def test_an_empty_annotation_still_arrives_as_answered(catalog, stocked, schema):
    ids, label_set_id = stocked
    catalog.annotations.annotate(ids[0], label_set_id, Choices())
    samples = catalog.labelled(label_set_id, EVERYTHING)
    [task] = build_tasks(samples, catalog, label_set_id, schema, ADDRESSING)

    # An empty result list is the answer "none of these apply", and it has
    # to arrive as an answer — otherwise rebuilding a project puts every
    # such sample back in the queue for someone to answer again
    assert task.annotations == []
    assert task.answered
    assert task.as_import()["annotations"] == [{"result": []}]


def test_a_skipped_sample_has_no_annotation_to_carry(catalog, stocked, schema):
    ids, label_set_id = stocked
    catalog.annotations.skip(ids[0], label_set_id)
    tasks = build_tasks(
        catalog.unlabelled(label_set_id, EVERYTHING), catalog, label_set_id, schema, ADDRESSING
    )
    assert all(t.sample_id != ids[0] for t in tasks)
