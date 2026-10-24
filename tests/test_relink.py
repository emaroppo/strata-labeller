"""Repointing tasks at new image URLs.

No Label Studio: `relink` works out what should change and hands the caller
a list, which is what makes a dry run free and an interrupted apply
describable.
"""

import pytest

from strata.catalog import EVERYTHING, SignedUrls
from strata.contracts import ClassificationSchema
from strata.labeller.labelstudio.adapter import TYPE_KEY, Addressing, blob_url
from strata.labeller.labelstudio.sync import relink

SECRET = "shared with the server"

LOCAL = Addressing(prefix="blobs")
SERVED = Addressing(prefix="blobs", urls=SignedUrls("http://minipc:8081", SECRET))


@pytest.fixture
def stocked(catalog, files):
    paths = files(3)
    catalog.ingest(paths, media="image", metadata_for=lambda p: {"source_path": str(p)})
    label_set_id = catalog.label_sets.create("presence", ClassificationSchema(classes=["cat"]))
    return catalog, catalog.samples.unlabelled(label_set_id, EVERYTHING)


def task(task_id: int, url: str, **extra) -> dict:
    return {"id": task_id, "data": {"image": url, **extra}}


def test_a_local_task_moves_to_the_serving_api(stocked):
    catalog, samples = stocked
    tasks = [task(1, blob_url(samples[0], "blobs"))]

    report = relink(tasks, catalog, SERVED, "image")

    assert len(report.changes) == 1
    task_id, data = report.changes[0]
    assert task_id == 1
    assert data["image"].startswith("http://minipc:8081/blob/")
    assert samples[0].checksum in data["image"]


def test_other_data_keys_survive(stocked):
    catalog, samples = stocked
    tasks = [task(1, blob_url(samples[0], "blobs"), note="looked odd")]

    _, data = relink(tasks, catalog, SERVED, "image").changes[0]
    # The API replaces data whole, so anything not carried forward is lost
    assert data["note"] == "looked odd"


def test_a_task_already_current_is_left_alone(stocked):
    catalog, samples = stocked
    tasks = [{"id": 1, "data": LOCAL.data_for(samples[0], "image")}]

    report = relink(tasks, catalog, LOCAL, "image")

    assert report.changes == []
    assert report.unchanged == 1


def test_a_task_from_before_types_gains_its_samples_type(stocked):
    catalog, samples = stocked
    url = LOCAL.url_for(samples[0])

    report = relink([task(1, url, note="looked odd")], catalog, LOCAL, "image")

    # The URL was already current; the type is what changes, so the task
    # keeps resolving once a second type holds the same bytes. docs/adr/0052
    assert report.changes == [(1, {"image": url, "note": "looked odd", TYPE_KEY: "image/plain"})]


def test_a_task_from_before_types_whose_bytes_two_samples_hold_is_left_alone(stocked, tmp_path):
    catalog, samples = stocked
    shared = samples[0].checksum
    # A second sample over the same bytes, of another type
    cached = catalog.ensure_cached([shared], tmp_path / "held")
    catalog.ingest([cached[shared]], media="image", subtype="copy")
    tasks = [
        task(1, blob_url(samples[0], "blobs")),
        {"id": 2, "data": LOCAL.data_for(samples[0], "image")},
    ]

    report = relink(tasks, catalog, SERVED, "image")

    # Giving it either type would be a guess at what the reviewer saw
    assert report.ambiguous == [1]
    assert [task_id for task_id, _ in report.changes] == [2]
    assert report.total == 2


def test_a_url_naming_no_sample_is_reported_not_rewritten(stocked):
    catalog, _ = stocked
    tasks = [task(1, "/data/local-files/?d=images/vid1/f001.jpg")]

    report = relink(tasks, catalog, SERVED, "image")

    # It shows a real image this cannot identify; rewriting it would destroy
    # the only record of what the reviewer was looking at
    assert report.changes == []
    assert len(report.unrecognised) == 1


def test_a_served_task_re_signs(stocked, monkeypatch):
    catalog, samples = stocked
    stale = SERVED.url_for(samples[0])

    # A queue that sat long enough for the signature to move on
    import strata.catalog.storage.signing as signing

    monkeypatch.setattr(signing, "window_expiry", lambda ttl=0: 9_999_999_999)
    report = relink([task(1, stale)], catalog, SERVED, "image")

    # The whole reason this is re-runnable: an expired link is a task that
    # will not load, and nothing else fixes it
    assert len(report.changes) == 1
    assert report.changes[0][1]["image"] != stale


def test_the_report_accounts_for_every_task(stocked):
    catalog, samples = stocked
    tasks = [
        task(1, blob_url(samples[0], "blobs")),
        task(2, SERVED.url_for(samples[1])),
        task(3, "/data/local-files/?d=images/old.jpg"),
    ]
    report = relink(tasks, catalog, SERVED, "image")
    assert report.total == 3
