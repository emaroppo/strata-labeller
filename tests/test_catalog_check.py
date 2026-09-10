"""Which catalog each machine is on.

Every machine reads its own config.toml, so pointing the setup at another
catalog is an edit on each. A machine missed does not fail loudly: a blob
server on the old catalog answers 404 for every new sample, and a modelling
host on it refuses every round. `catalog-check` asks each one.
"""

import json
import urllib.error
import urllib.request

import pytest
from typer.testing import CliRunner

from strata.catalog import Catalog
from strata.labeller.cli import app
from strata.modelling.service import PROTOCOL

ELSEWHERE = "20250101T000000-cccccccc"


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
def setup(tmp_path, monkeypatch):
    """A catalog here, a blob server and a modelling host, each answering /healthz."""
    catalog = Catalog.local(tmp_path / "catalog")
    config = tmp_path / "config.toml"
    config.write_text(
        f'[catalog]\nroot = "{tmp_path / "catalog"}"\nserve_url = "http://blobs:8081"\n\n'
        f'[modelling]\nurl = "http://gpu:8082"\n'
    )
    answers = {"blobs": catalog.id, "gpu": catalog.id}

    def fake_urlopen(request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        host = "blobs" if "blobs:8081" in url else "gpu"
        payload = {"ok": True, "catalog": {"name": "default", "id": answers[host]}}
        if host == "gpu":
            payload["protocol"] = PROTOCOL
        return Reply(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return config, answers


def _check(config):
    return CliRunner().invoke(app, ["catalog-check", "--config", str(config)])


def test_every_machine_on_this_catalog_passes(setup):
    config, _ = setup
    result = _check(config)
    assert result.exit_code == 0, result.output
    assert result.output.count("same catalog") == 2


def test_a_machine_left_on_another_catalog_fails_naming_it(setup):
    config, answers = setup
    answers["blobs"] = ELSEWHERE
    result = _check(config)
    assert result.exit_code == 1
    assert "blob server" in result.output
    assert ELSEWHERE in result.output


def test_an_index_that_wants_a_password_says_so(tmp_path, monkeypatch):
    """Not a traceback: the likeliest reason, on a machine that just switched."""
    from strata.labeller import cli

    def refuse(settings, name=""):
        raise RuntimeError(
            'connection failed: connection to server at "db", port 5432 failed: '
            "fe_sendauth: no password supplied"
        )

    monkeypatch.setattr(cli, "_catalog_if_any", refuse)
    monkeypatch.delenv("PGPASSWORD", raising=False)
    config = tmp_path / "config.toml"
    config.write_text('[catalog]\nurl = "postgresql+psycopg://strata@db:5432/strata"\n')

    result = _check(config)

    assert result.exit_code == 1
    # The console wraps long lines, so compare with the wrapping taken out
    said = " ".join(result.output.split())
    assert "PGPASSWORD is not set" in said
    assert "Traceback" not in said


def test_an_unreachable_machine_fails(setup, monkeypatch):
    config, _ = setup

    def refuse(request, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    result = _check(config)
    assert result.exit_code == 1
    assert "unreachable" in result.output
