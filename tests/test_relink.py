"""Repointing tasks at new image URLs.

No Label Studio: `relink` works out what should change and hands the caller
a list, which is what makes a dry run free and an interrupted apply
describable.
"""

import pytest

from strata.catalog import EVERYTHING, SignedUrls
from strata.labeller.labelstudio.adapter import Addressing, blob_url
from strata.labeller.labelstudio.sync import relink
from strata.labels import ClassificationSchema

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
    tasks = [task(1, LOCAL.url_for(samples[0]))]

    report = relink(tasks, catalog, LOCAL, "image")

    assert report.changes == []
    assert report.unchanged == 1


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
