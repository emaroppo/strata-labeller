"""More than one catalog on a host, and which one a command opens.

A host used to talk to exactly one catalog because `config.toml` held one
`[catalog]` table. That is fine until a second job wants a different corpus
— different media, different collections, different bucket — and the only
way to have one was a second config file and the discipline to pass
`--config` correctly every time.

Everything here is about the same failure: sample ids mean nothing outside
the catalog that issued them, so a command that opens the wrong one reports
real numbers about the wrong data, and labels real samples against the
wrong ids. Nothing raises. So the rule throughout is that an unresolvable
name is refused rather than guessed.
"""

import pytest

from strata.labeller.config import DEFAULT_CATALOG, ConfigError, Settings


def _write(tmp_path, body: str):
    path = tmp_path / "config.toml"
    path.write_text(body)
    return path


# ----------------------------------------------------------------------
# A host with one catalog, which is every host that existed before this
# ----------------------------------------------------------------------


def test_a_flat_catalog_still_works(tmp_path):
    settings = Settings.load(_write(tmp_path, '[catalog]\nroot = "somewhere"\n'))
    assert settings.catalog.root == "somewhere"
    assert settings.catalog_names() == [DEFAULT_CATALOG]
    # The same object by either route, so a caller that never heard of
    # names is reading the catalog it always read
    assert settings.catalog_named() is settings.catalog


def test_no_catalog_section_at_all(tmp_path):
    settings = Settings.load(_write(tmp_path, "[label_studio]\nurl = \"x\"\n"))
    assert settings.catalog_named().root == "catalog"


# ----------------------------------------------------------------------
# Named catalogs
# ----------------------------------------------------------------------


def test_named_catalogs_are_addressable(tmp_path):
    settings = Settings.load(_write(tmp_path, """
[catalog.images]
root = "images"

[catalog.text]
root = "text"
"""))
    assert settings.catalog_names() == ["images", "text"]
    assert settings.catalog_named("images").root == "images"
    assert settings.catalog_named("text").root == "text"


def test_host_settings_are_stated_once(tmp_path):
    """Scalars are the host's, tables layer on top.

    Two catalogs in one bucket is the ordinary case — same machine, same
    storage, different corpora — and repeating the endpoint, region and
    secret in every table is how one of them ends up subtly different.
    """
    settings = Settings.load(_write(tmp_path, """
[catalog]
s3_endpoint = "http://garage:3900"
s3_bucket = "shared"

[catalog.images]
root = "images"

[catalog.text]
root = "text"
s3_bucket = "text-only"
"""))
    assert settings.catalog_named("images").s3_endpoint == "http://garage:3900"
    assert settings.catalog_named("text").s3_endpoint == "http://garage:3900"
    # and a table overrides what it names
    assert settings.catalog_named("images").s3_bucket == "shared"
    assert settings.catalog_named("text").s3_bucket == "text-only"


def test_a_single_named_catalog_needs_no_default(tmp_path):
    settings = Settings.load(_write(tmp_path, '[catalog.only]\nroot = "one"\n'))
    assert settings.default_catalog == "only"
    assert settings.catalog.root == "one"


def test_an_explicit_default_is_honoured(tmp_path):
    settings = Settings.load(_write(tmp_path, """
[catalog]
default = "text"

[catalog.images]
root = "images"

[catalog.text]
root = "text"
"""))
    assert settings.catalog.root == "text"


# ----------------------------------------------------------------------
# What is refused
# ----------------------------------------------------------------------


def test_an_unknown_name_lists_the_ones_that_exist(tmp_path):
    settings = Settings.load(_write(tmp_path, """
[catalog.images]
root = "images"

[catalog.text]
root = "text"
"""))
    with pytest.raises(ConfigError, match="images, text"):
        settings.catalog_named("satellite")


def test_an_unknown_name_does_not_fall_back(tmp_path):
    """The whole point. A typo must not silently open another corpus."""
    settings = Settings.load(_write(tmp_path, '[catalog.images]\nroot = "images"\n'))
    with pytest.raises(ConfigError):
        settings.catalog_named("imagse")


def test_several_catalogs_and_no_default_is_ambiguous(tmp_path):
    settings = Settings.load(_write(tmp_path, """
[catalog.images]
root = "images"

[catalog.text]
root = "text"
"""))
    # Not at load time — a project or --catalog naming one settles it, and
    # only asking for "the default" is genuinely ambiguous
    with pytest.raises(ConfigError, match="several catalogs"):
        settings.catalog_named()
    assert settings.catalog_named("text").root == "text"


def test_a_default_naming_nothing_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="names no catalog"):
        Settings.load(_write(tmp_path, """
[catalog]
default = "missing"

[catalog.images]
root = "images"
"""))


def test_an_unknown_key_says_where_it_is(tmp_path):
    with pytest.raises(ConfigError, match="catalog.images"):
        Settings.load(_write(tmp_path, '[catalog.images]\nrooot = "typo"\n'))


# ----------------------------------------------------------------------
# The environment
# ----------------------------------------------------------------------


def test_credentials_reach_every_catalog(tmp_path, monkeypatch):
    """They describe the host's storage, not one corpus in it."""
    monkeypatch.setenv("STRATA_S3_ACCESS_KEY", "key")
    monkeypatch.setenv("STRATA_BLOB_SECRET", "secret")
    settings = Settings.load(_write(tmp_path, """
[catalog.images]
root = "images"

[catalog.text]
root = "text"
"""))
    for name in ("images", "text"):
        assert settings.catalog_named(name).s3_access_key == "key"
        assert settings.catalog_named(name).blob_secret == "secret"


def test_a_catalog_url_lands_on_the_default_only(tmp_path, monkeypatch):
    """It names one index, and two names for one database is a lie.

    Merging relies on identity, and two catalogs sharing a database share
    an identity — so this override applying to all of them would make the
    host claim two corpora where it has one.
    """
    monkeypatch.setenv("STRATA_CATALOG_URL", "postgresql://host/db")
    settings = Settings.load(_write(tmp_path, """
[catalog]
default = "images"

[catalog.images]
root = "images"

[catalog.text]
root = "text"
"""))
    assert settings.catalog.url == "postgresql://host/db"
    assert settings.catalog_named("text").url == ""


# ----------------------------------------------------------------------
# What a project asks for
# ----------------------------------------------------------------------


def test_a_project_names_the_catalog_it_draws_from(tmp_path):
    from strata.labeller.project import Project

    root = tmp_path / "job"
    root.mkdir()
    (root / "project.toml").write_text("""
[label_config]
classes = ["a"]

[data]
type = "image"

[catalog]
name = "images"
label_set = "job"
""")
    assert Project.load(root).catalog.name == "images"


def test_a_project_that_names_none_gets_the_default(tmp_path):
    from strata.labeller.project import Project

    root = tmp_path / "job"
    root.mkdir()
    (root / "project.toml").write_text(
        '[label_config]\nclasses = ["a"]\n\n[data]\ntype = "image"\n'
    )
    # Empty, not a guess: what "the default" means is the host's business,
    # so a project written before any of this still loads and still works
    assert Project.load(root).catalog.name == ""


# ----------------------------------------------------------------------
# End to end: two jobs, two corpora, one host
# ----------------------------------------------------------------------


def test_two_projects_ingest_into_their_own_catalogs(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from strata.catalog import Catalog
    from strata.labeller.cli import app

    monkeypatch.chdir(tmp_path)
    config = _write(tmp_path, f"""
[catalog]
default = "images"

[catalog.images]
root = "{tmp_path / 'images'}"

[catalog.text]
root = "{tmp_path / 'text'}"
""")

    runner = CliRunner()
    for job, catalog_name, count in (("cats", "images", 2), ("notes", "text", 3)):
        root = tmp_path / job
        (root / "data" / "raw").mkdir(parents=True)
        (root / "project.toml").write_text(
            f'[label_config]\nclasses = ["a"]\n\n[data]\ntype = "image"\n\n'
            f'[catalog]\nname = "{catalog_name}"\n'
        )
        for i in range(count):
            (root / "data" / "raw" / f"{i}.jpg").write_bytes(f"{job} {i}".encode())
        result = runner.invoke(app, ["ingest", "-p", str(root), "--config", str(config)])
        assert result.exit_code == 0, result.stdout

    images, text = Catalog.local(tmp_path / "images"), Catalog.local(tmp_path / "text")
    # Separate corpora, and separate identities — which is what lets a run,
    # a merge and a task map each say which one they belong to
    assert images.id != text.id
    assert len(images.unlabelled(images.label_set("cats")[0], "*")) == 2
    assert len(text.unlabelled(text.label_set("notes")[0], "*")) == 3
