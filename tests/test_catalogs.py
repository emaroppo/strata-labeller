"""More than one catalog on a host, and which one a command opens.

How a catalog is described, and which is the default, is tested where the
parsing lives: ``packages/catalog/tests/test_catalog_config.py``. What stays here is
the labeller's side of it — that its settings read the catalogs through
that code, that a project names the one it draws from, and that two jobs on
one host each reach their own.

The failure all of this guards against is the same: sample ids mean nothing
outside the catalog that issued them, so a command that opens the wrong one
reports real numbers about the wrong data, and labels real samples against
the wrong ids. Nothing raises.
"""

from strata.labeller.config import Settings


def _write(tmp_path, body: str):
    path = tmp_path / "config.toml"
    path.write_text(body)
    return path


# ----------------------------------------------------------------------
# The labeller's settings
# ----------------------------------------------------------------------


def test_settings_read_the_catalogs_the_file_describes(tmp_path):
    settings = Settings.load(_write(tmp_path, """
[catalog]
default = "text"

[catalog.images]
root = "images"

[catalog.text]
root = "text"
"""))
    assert settings.catalogs.named().root == "text"
    assert settings.catalogs.named("images").root == "images"


def test_task_urls_use_each_catalogs_own_mount(tmp_path):
    """One catalog's tasks read off `blobs`, another's off `blobs-emails`, in one Label Studio."""
    from strata.labeller.cli._shared import _addressing

    settings = Settings.load(_write(tmp_path, """
[catalog.main]
root = "main"

[catalog.emails]
root = "emails"
blobs_prefix = "blobs-emails"
"""))
    assert _addressing(settings, settings.catalogs.named("emails")).prefix == "blobs-emails"
    assert _addressing(settings, settings.catalogs.named("main")).prefix == "blobs"


# ----------------------------------------------------------------------
# What a project asks for
# ----------------------------------------------------------------------


def test_a_project_names_the_catalog_it_draws_from(tmp_path):
    from strata.labeller.project import LabellingProject

    root = tmp_path / "job"
    root.mkdir()
    (root / "project.toml").write_text("""
[label_set]
classes = ["a"]

[data]
type = "image"

[catalog]
name = "images"
label_set = "job"
""")
    assert LabellingProject.load(root).catalog.name == "images"


def test_a_project_that_names_none_gets_the_default(tmp_path):
    from strata.labeller.project import LabellingProject

    root = tmp_path / "job"
    root.mkdir()
    (root / "project.toml").write_text(
        '[label_set]\nclasses = ["a"]\n\n[data]\ntype = "image"\n'
    )
    # Empty, not a guess: what "the default" means is the host's business,
    # so a project written before any of this still loads and still works
    assert LabellingProject.load(root).catalog.name == ""


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
            f'[label_set]\nclasses = ["a"]\n\n[data]\ntype = "image"\n\n'
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
    assert len(images.samples.unlabelled(images.label_sets.get("cats")[0], "*")) == 2
    assert len(text.samples.unlabelled(text.label_sets.get("notes")[0], "*")) == 3
