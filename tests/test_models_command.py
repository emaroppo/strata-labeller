"""What the machine that trains can serve, asked before a round is.

The backend is the only authority on what it can serve, and the round is
checked when it arrives. What was missing was anything asking earlier: a
project pointed at a model the host does not have found out at its first
train, after the labelling.
"""

import json
import urllib.request

import pytest
from typer.testing import CliRunner

from strata.labeller.cli import app
from strata.modelling.remote.wire import PROTOCOL

SERVED = {"multilabel": "strata.modelling.baselines:MultiLabelClassifier"}


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
def host(tmp_path, monkeypatch):
    """A modelling host answering /healthz and /models."""
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.toml"
    config.write_text(
        f'[catalog]\nroot = "{tmp_path / "catalog"}"\n\n[modelling]\nurl = "http://gpu:8082"\n'
    )

    def fake_urlopen(request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        if url.endswith("/healthz"):
            return Reply({"ok": True, "protocol": PROTOCOL, "catalog": {"name": "d", "id": "c"}})
        assert url.endswith("/models"), url
        return Reply({"models": SERVED})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return config


def _models(config, *args):
    return CliRunner().invoke(app, ["models", "--config", str(config), *args])


def test_the_hosts_list_is_what_it_serves(host):
    result = _models(host)
    assert result.exit_code == 0, result.output
    assert "modelling host" in result.output and "multilabel" in result.output

    payload = json.loads(_models(host, "--json").output)
    assert payload["where"] == "modelling host" and payload["models"] == SERVED


def test_a_projects_registered_model_is_reported_served(host, make_project):
    project = make_project("demo", classes=["cat"])
    result = _models(host, "-p", str(project.root))
    assert result.exit_code == 0, result.output
    assert "served" in result.output


def test_a_projects_file_reference_cannot_run_on_a_host(host, make_project):
    project = make_project("demo", classes=["cat"])
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace('ref = "multilabel"', 'ref = "model.py:Toy"'))

    result = _models(host, "-p", str(project.root))

    assert result.exit_code == 1
    # The console wraps long lines, so compare with the wrapping taken out
    said = " ".join(result.output.split())
    assert "file reference" in said and "not on a host" in said


def test_without_a_host_the_list_is_this_machines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.toml"
    config.write_text(f'[catalog]\nroot = "{tmp_path / "catalog"}"\n')

    result = _models(config)

    assert result.exit_code == 0, result.output
    assert "this machine" in result.output and "multilabel" in result.output


@pytest.fixture
def two_hosts(tmp_path, monkeypatch):
    """Two modelling hosts serving different models; the default is gpu."""
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.toml"
    config.write_text(
        f'[catalog]\nroot = "{tmp_path / "catalog"}"\n\n'
        '[modelling]\ndefault = "gpu"\n\n'
        '[modelling.gpu]\nurl = "http://gpu:8082"\n\n'
        '[modelling.gx10]\nurl = "http://gx10:8082"\n'
    )
    served = {"gpu": SERVED, "gx10": {"spans": "strata.modelling.baselines:Spans"}}

    def fake_urlopen(request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        if url.endswith("/healthz"):
            return Reply({"ok": True, "protocol": PROTOCOL, "catalog": {"name": "d", "id": "c"}})
        return Reply({"models": served["gx10" if "gx10" in url else "gpu"]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return config


def test_host_asks_the_host_it_names(two_hosts):
    payload = json.loads(_models(two_hosts, "--json", "--host", "gx10").output)
    assert payload["where"] == "modelling host gx10"
    assert payload["url"] == "http://gx10:8082" and "spans" in payload["models"]


def test_a_project_naming_a_host_is_asked_against_that_host(two_hosts, make_project):
    project = make_project("demo", classes=["cat"])
    toml = project.root / "project.toml"
    toml.write_text(
        toml.read_text().replace('ref = "multilabel"', 'ref = "multilabel"\nhost = "gx10"')
    )

    result = _models(two_hosts, "-p", str(project.root))

    # gx10 serves no multilabel, though the default host does
    assert result.exit_code == 1
    assert "gx10" in result.output


def test_an_unknown_host_is_refused_naming_the_ones_there_are(two_hosts):
    result = _models(two_hosts, "--host", "dgx")
    assert result.exit_code == 1
    assert "gpu, gx10" in result.output


def test_a_projects_catalog_chooses_the_host_when_nothing_else_does(two_hosts, make_project):
    toml = two_hosts.read_text().replace(
        '[modelling.gx10]\nurl = "http://gx10:8082"\n',
        '[modelling.gx10]\nurl = "http://gx10:8082"\ndefault_for = ["default"]\n',
    )
    two_hosts.write_text(toml)
    project = make_project("demo", classes=["cat"])

    payload = json.loads(_models(two_hosts, "--json", "-p", str(project.root)).output)

    # The machine's default is gpu; the catalog's is gx10, and wins
    assert payload["url"] == "http://gx10:8082"
