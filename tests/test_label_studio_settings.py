"""A catalog can be labelled on a Label Studio of its own. docs/adr/0044"""

import pytest

from strata.catalog.config import CatalogConfigError
from strata.labeller.config import Settings

BODY = """
[label_studio]
url = "http://localhost:8080"

[label_studio.main]
url = "http://minipc:8090"

[catalog]
default = "main"

[catalog.main]
root = "main"

[catalog.local]
root = "local"
"""


def _load(tmp_path, body=BODY, environ=None):
    path = tmp_path / "config.toml"
    path.write_text(body)
    return Settings.load(path, environ=environ or {})


def test_a_catalog_with_its_own_instance_is_labelled_there(tmp_path):
    settings = _load(tmp_path)
    assert settings.label_studio_for("main").url == "http://minipc:8090"
    assert settings.label_studio_for("").url == "http://minipc:8090"
    assert settings.label_studio_for("local").url == "http://localhost:8080"


def test_flat_keys_are_shared_by_each_instance(tmp_path):
    body = BODY.replace("[label_studio]\n", '[label_studio]\nlocal_storage_path = "/x"\n')
    assert _load(tmp_path, body).label_studio_for("main").local_storage_path == "/x"


def test_an_instances_key_is_its_own_and_never_the_machines(tmp_path):
    settings = _load(tmp_path, environ={"LABEL_STUDIO_API_KEY": "laptop"})
    assert settings.label_studio_for("local").api_key == "laptop"
    assert settings.label_studio_for("main").api_key == ""
    assert settings.label_studio_for("main").key_from() == "$LABEL_STUDIO_API_KEY_MAIN"

    environ = {"LABEL_STUDIO_API_KEY": "laptop", "LABEL_STUDIO_API_KEY_MAIN": "minipc"}
    assert _load(tmp_path, environ=environ).label_studio_for("main").api_key == "minipc"


def test_an_instance_for_an_unknown_catalog_is_refused(tmp_path):
    with pytest.raises(CatalogConfigError, match="mian"):
        _load(tmp_path, BODY.replace("[label_studio.main]", "[label_studio.mian]"))


def test_an_unknown_key_is_refused(tmp_path):
    with pytest.raises(CatalogConfigError, match=r"label_studio\.main"):
        _load(tmp_path, BODY.replace('url = "http://minipc:8090"', "port = 8090"))


def test_a_flat_section_alone_is_the_machines_instance(tmp_path):
    settings = _load(tmp_path, '[label_studio]\nurl = "http://ls:8080"\n')
    assert settings.label_studio_for("").url == "http://ls:8080"
