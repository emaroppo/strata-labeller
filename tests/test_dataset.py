"""Dataset load/save, including the v1 format still on disk in old projects."""

import json

import pytest

from strata.labeller.dataset import (
    DATASET_VERSION,
    Sample,
    get_classes,
    is_v1,
    load_dataset,
    save_dataset,
    split_labeled_unlabeled,
    train_val_split,
)
from strata.labeller.schemas import ClassificationSchema


@pytest.fixture
def schema():
    return ClassificationSchema(["cat", "dog"])


def write(path, payload):
    path.write_text(json.dumps(payload))
    return path


# ----------------------------------------------------------------------
# v1 upgrade on read
# ----------------------------------------------------------------------


def test_v1_labels_become_results_and_count_as_annotated(tmp_path, schema):
    path = write(
        tmp_path / "dataset.json",
        [{"path": "a.jpg", "labels": ["cat"]}, {"path": "b.jpg"}],
    )
    a, b = load_dataset(path, schema)

    assert a.annotated is True
    assert schema.decode_target(a.results) == ["cat"]
    # No labels in v1 meant nobody had looked
    assert b.annotated is False
    assert b.results == []


def test_v1_upgrade_needs_a_schema_only_when_there_are_labels(tmp_path, schema):
    unlabeled = write(tmp_path / "empty.json", [{"path": "a.jpg"}])
    assert load_dataset(unlabeled, schema=None)[0].path == "a.jpg"

    labeled = write(tmp_path / "labeled.json", [{"path": "a.jpg", "labels": ["cat"]}])
    with pytest.raises(ValueError, match="needs the project's schema"):
        load_dataset(labeled, schema=None)


def test_v1_carries_the_skipped_flag_across(tmp_path, schema):
    path = write(tmp_path / "dataset.json", [{"path": "a.jpg", "skipped": True}])
    assert load_dataset(path, schema)[0].skipped is True


def test_is_v1_detects_the_bare_list(tmp_path):
    assert is_v1(write(tmp_path / "old.json", [{"path": "a.jpg"}])) is True
    assert is_v1(write(tmp_path / "new.json", {"version": 2, "samples": []})) is False


# ----------------------------------------------------------------------
# v2
# ----------------------------------------------------------------------


def test_v2_round_trip(tmp_path, schema):
    samples = [
        Sample(path="a.jpg", results=schema.encode_target(["cat"]), annotated=True),
        Sample(path="b.jpg"),
        Sample(path="c.jpg", skipped=True),
    ]
    path = tmp_path / "dataset.json"
    save_dataset(samples, path)
    assert load_dataset(path, schema) == samples


def test_an_annotation_with_no_results_survives_as_annotated(tmp_path, schema):
    # An image a human looked at and found nothing in is a real annotation;
    # only the flag distinguishes it from one nobody has opened
    path = tmp_path / "dataset.json"
    save_dataset([Sample(path="a.jpg", annotated=True)], path)

    stored = json.loads(path.read_text())["samples"][0]
    assert stored == {"path": "a.jpg", "annotated": True}
    assert load_dataset(path, schema)[0].annotated is True


def test_saving_omits_the_flags_that_are_false(tmp_path):
    path = tmp_path / "dataset.json"
    save_dataset([Sample(path="a.jpg")], path)
    assert json.loads(path.read_text()) == {
        "version": DATASET_VERSION,
        "samples": [{"path": "a.jpg"}],
    }


def test_saving_creates_the_parent_directory(tmp_path):
    path = tmp_path / "nested" / "deeper" / "dataset.json"
    save_dataset([Sample(path="a.jpg")], path)
    assert path.exists()


def test_an_unknown_version_is_refused_rather_than_guessed(tmp_path, schema):
    path = write(tmp_path / "dataset.json", {"version": 99, "samples": []})
    with pytest.raises(ValueError, match="dataset version 99"):
        load_dataset(path, schema)


# ----------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------


def test_skipped_samples_belong_to_neither_pool():
    samples = [
        Sample(path="labeled.jpg", annotated=True),
        Sample(path="fresh.jpg"),
        Sample(path="skipped.jpg", skipped=True),
        # Reviewed, skipped, and annotated: still out of both pools
        Sample(path="both.jpg", annotated=True, skipped=True),
    ]
    labeled, unlabeled = split_labeled_unlabeled(samples)
    assert [s.path for s in labeled] == ["labeled.jpg"]
    assert [s.path for s in unlabeled] == ["fresh.jpg"]


def test_train_val_split_is_deterministic_for_a_seed():
    samples = [Sample(path=f"{i}.jpg") for i in range(20)]
    first = train_val_split(samples, seed=7)
    second = train_val_split(samples, seed=7)
    assert [s.path for s in first[0]] == [s.path for s in second[0]]
    assert len(first[0]) + len(first[1]) == 20


def test_a_flat_folder_splits_by_ratio_without_a_group_key():
    # The layout the README documents. Grouping these by folder would put
    # them all in one group and leave nothing to train on, which is why an
    # "images" project passes no group key at all.
    samples = [Sample(path=f"{i}.jpg") for i in range(50)]
    train, val = train_val_split(samples, val_ratio=0.2)
    assert (len(train), len(val)) == (40, 10)


def test_group_key_keeps_a_group_on_one_side():
    # All frames of one video must land together, or near-duplicates leak
    # from train into val and the metric flatters itself
    samples = [Sample(path=f"vid{v}/frame{f}.jpg") for v in range(6) for f in range(4)]
    train, val = train_val_split(samples, val_ratio=0.5, group_key=lambda s: s.path.split("/")[0])

    def videos(bucket):
        return {s.path.split("/")[0] for s in bucket}

    assert videos(train) & videos(val) == set()
    assert len(train) + len(val) == len(samples)


def test_get_classes_reads_through_the_schema(schema):
    samples = [
        Sample(path="a.jpg", results=schema.encode_target(["dog"]), annotated=True),
        Sample(path="b.jpg", results=schema.encode_target(["cat", "dog"]), annotated=True),
    ]
    assert get_classes(samples, schema) == ["cat", "dog"]
