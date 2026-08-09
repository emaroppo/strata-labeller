"""Every command gets far enough to fail on its own terms.

Commands do their importing inside the function body, so a module that
moves between packages breaks the command and nothing else — the suite goes
on passing because no test invokes it, and the failure surfaces on whichever
machine runs it first.

That has now happened twice to `push`. These tests assert almost nothing
about behaviour: only that a command reaches its own logic rather than dying
on an import. They fail on a missing catalog or a missing token, which is
the point — those are real answers, and ImportError is not.
"""

import pytest
from typer.testing import CliRunner

from strata.labeller.cli import app

runner = CliRunner()

#: Everything that takes a project. The rest either scaffold or take
#: explicit arguments, and are covered where their behaviour is.
COMMANDS = [
    ["push"],
    ["export"],
    ["train"],
    ["init"],
    ["ingest"],
    ["relink"],
    ["report"],
    ["unskip", "--all"],
    ["class", "list"],
    ["catalog-stats"],
]


@pytest.fixture
def bare(make_project, tmp_path):
    """A project with nothing set up around it, and a config to match."""
    project = make_project("demo")
    config = tmp_path / "config.toml"
    config.write_text(f'[catalog]\nroot = "{tmp_path / "catalog"}"\n')
    return project, config


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: " ".join(c))
def test_a_command_fails_on_its_own_terms(command, bare):
    project, config = bare
    result = runner.invoke(
        app, [*command, "-p", str(project.root), "--config", str(config)]
    )

    exception = result.exception
    if isinstance(exception, ModuleNotFoundError | ImportError | AttributeError):
        raise AssertionError(
            f"'{' '.join(command)}' did not reach its own logic: "
            f"{type(exception).__name__}: {exception}"
        )
