"""Scaffolding a project: where it lands, and what it says to do next."""

import pytest
from typer.testing import CliRunner

from strata.labeller.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def elsewhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def new(*args):
    result = runner.invoke(app, ["new", *args, "--class", "a"])
    assert result.exit_code == 0, result.output
    return result.output


def test_a_bare_name_lands_under_projects(elsewhere):
    new("cats")
    assert (elsewhere / "projects" / "cats" / "project.toml").exists()


@pytest.mark.parametrize("typed", ["./cats", "work/cats"])
def test_anything_path_shaped_is_taken_literally(elsewhere, typed):
    # Path("./cats") has already dropped its "./", so this is read off what
    # was typed rather than off the path
    new(typed)
    assert (elsewhere / typed / "project.toml").exists()
    assert not (elsewhere / "projects").exists()


def test_an_absolute_path_is_taken_literally(elsewhere):
    target = elsewhere / "abs" / "cats"
    new(str(target))
    assert (target / "project.toml").exists()


def test_the_next_step_is_preparing_then_ingesting(elsewhere):
    # Ingest refuses a data root with no prepared index, so saying only
    # "ingest" sends a new user straight into that refusal. docs/adr/0040
    output = " ".join(new("cats").split())
    assert "--preparer image-folder" in output
    assert output.index("prepare") < output.index("strata-labeller ingest")
