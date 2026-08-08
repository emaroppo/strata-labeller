"""Talking to a modelling host.

No server: what is under test is the client's half — what it sends, and
what it does with what comes back. The failures matter more than the happy
path, because a round is minutes long and an unclear failure at the end of
one is expensive.
"""

import json
import urllib.error
import urllib.request

import pytest

from strata.labeller.remote import Refused, Trainer, Unreachable


class Reply:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def sent(monkeypatch):
    """Capture the request instead of making it."""
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data) if request.data else None
        captured["timeout"] = timeout
        return Reply(captured.get("reply", {"run": {"id": 1}, "metrics": {}}))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


def test_a_round_sends_an_id_not_a_dataset(sent):
    Trainer("http://gpu:8082", "t").submit(7, "multilabel", {"num_epochs": 4})

    assert sent["url"] == "http://gpu:8082/round"
    # The host has the index and the bucket; sending it images it can fetch
    # itself would be paying the network to avoid the network
    assert sent["body"] == {
        "dataset_id": 7,
        "model": "multilabel",
        "params": {"num_epochs": 4},
        "fresh": False,
    }


def test_the_token_travels(sent):
    Trainer("http://gpu:8082", "sekrit").submit(7, "multilabel", {})
    assert sent["headers"]["Authorization"] == "Bearer sekrit"


def test_a_trailing_slash_does_not_double_up(sent):
    Trainer("http://gpu:8082/", "t").submit(7, "m", {})
    assert sent["url"] == "http://gpu:8082/round"


def test_submitting_is_a_short_request(sent):
    Trainer("http://gpu:8082", "t").submit(7, "m", {})
    # The round outlives the request that asked for it, so this waits for an
    # acknowledgement rather than for training
    assert sent["timeout"] <= 60


def test_listing_models_does_not_wait_for_hours(sent):
    sent["reply"] = {"models": {"multilabel": "x:Y"}}
    assert Trainer("http://gpu:8082", "t").models() == {"multilabel": "x:Y"}
    assert sent["timeout"] <= 60


def test_the_hosts_reason_is_what_surfaces(monkeypatch):
    def refuse(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 400, "Bad Request", {},
            _Body(json.dumps({"detail": "register it as an entry point, or run locally"})),
        )

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(Refused, match="entry point"):
        Trainer("http://gpu:8082", "t").submit(7, "model.py:Custom", {})


def test_an_unreachable_host_says_so(monkeypatch):
    def unreachable(request, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", unreachable)
    with pytest.raises(Unreachable, match="Could not reach"):
        Trainer("http://gpu:8082", "t").submit(7, "m", {})


class _Body:
    """HTTPError closes what it is handed, so this has to be closeable."""

    def __init__(self, text):
        self.text = text

    def read(self):
        return self.text.encode()

    def close(self):
        pass


# ----------------------------------------------------------------------
# Following a round from a network that comes and goes
# ----------------------------------------------------------------------


def test_following_ends_when_the_round_does(monkeypatch):
    states = [
        {"state": "running", "stage": "materialising", "done": 5, "total": 10},
        {"state": "running", "stage": "training"},
        {"state": "done", "result": {"run": {"id": 3}}},
    ]
    trainer = Trainer("http://gpu:8082", "t")
    monkeypatch.setattr(trainer, "job", lambda job_id: states.pop(0))

    seen = []
    final = trainer.follow("abc", on_state=seen.append, sleep=lambda _: None)

    assert final["result"]["run"]["id"] == 3
    assert [s["state"] for s in seen] == ["running", "running", "done"]


def test_a_dropped_network_does_not_end_a_round(monkeypatch):
    """The coffee shop case: the wifi fails, the training does not."""
    answers = [
        Unreachable("Could not reach"),
        Unreachable("Could not reach"),
        {"state": "done", "result": {"run": {"id": 3}}},
    ]

    def flaky(job_id):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    trainer = Trainer("http://gpu:8082", "t")
    monkeypatch.setattr(trainer, "job", flaky)

    seen = []
    final = trainer.follow("abc", on_state=seen.append, sleep=lambda _: None)

    assert final["state"] == "done"
    # Reported, not hidden — a silent stall looks like a hung round
    assert [s["state"] for s in seen] == ["unreachable", "unreachable", "done"]


def test_a_refusal_ends_it(monkeypatch):
    def refuse(job_id):
        raise Refused("no such job")

    trainer = Trainer("http://gpu:8082", "t")
    monkeypatch.setattr(trainer, "job", refuse)
    # The host answered. Asking again produces the same answer more often.
    with pytest.raises(Refused):
        trainer.follow("abc", sleep=lambda _: None)


def test_a_failed_round_is_returned_not_raised(monkeypatch):
    trainer = Trainer("http://gpu:8082", "t")
    monkeypatch.setattr(
        trainer, "job", lambda job_id: {"state": "failed", "error": "CUDA out of memory"}
    )
    # The caller wants the reason, and a reason is data rather than an
    # exception from the polling loop
    assert trainer.follow("abc", sleep=lambda _: None)["error"] == "CUDA out of memory"
