"""Talking to the modelling host and the blob server from a command."""

import json

import typer

from ._shared import _error, _progress, console


def _reattach(settings, job_id: str) -> None:
    """Pick up a round that is already running elsewhere."""
    from strata.modelling.remote.client import Trainer

    if not settings.modelling.url:
        _error("No modelling host configured, so there is no job to reattach to.")
        raise typer.Exit(1)

    trainer = Trainer(settings.modelling.url, settings.modelling.token)
    _print_run_result(_follow(trainer, job_id))


def _trainer(settings):
    """The modelling host's client, refused without the token it was started with."""
    from strata.modelling.remote.client import Trainer

    if not settings.modelling.token:
        _error(
            "No token for the modelling host. Set $STRATA_MODELLING_TOKEN to "
            "the same value it was started with."
        )
        raise typer.Exit(1)
    return Trainer(settings.modelling.url, settings.modelling.token)


def _print_run_result(result: dict) -> None:
    """A finished remote round: the run, its parent, and its metrics."""
    run = result["run"]
    console.print(
        f"[green]Run {run['id']}[/green]"
        + (f", continuing run {run['parent_run_id']}" if run.get("parent_run_id") else " (cold)")
    )
    if result.get("materialised"):
        console.print(f"  [dim]{result['materialised']:,} sample(s) materialised there[/dim]")
    for metric, value in sorted(result.get("metrics", {}).items()):
        console.print(f"  {metric}: {value}")


def _follow(trainer, job_id: str) -> dict:
    """Watch a round to its end, surviving a network that comes and goes.

    Interrupting this stops watching, not training. See ``docs/adr/0007``.
    """
    from strata.modelling.remote.client import RemoteError

    # Elapsed, because every stage looks identical while it is running
    # and the long one looks identical to a stalled one
    with _progress(elapsed=True, transient=True) as progress:
        bar = progress.add_task("Waiting for the host...")

        def show(job: dict) -> None:
            if job.get("state") == "unreachable":
                progress.update(
                    bar,
                    description=(
                        f"[yellow]Cannot reach the host (attempt "
                        f"{job['attempts']}) — the round is unaffected[/yellow]"
                    ),
                )
                return
            stage = job.get("stage", job.get("state", ""))
            if job.get("total"):
                stage = f"{stage} {job['done']:,}/{job['total']:,}"
            progress.update(bar, description=f"Host: {stage}")

        try:
            job = trainer.follow(job_id, on_state=show)
        except KeyboardInterrupt:
            progress.stop()
            console.print(
                f"[yellow]Stopped watching. The round is still running on the "
                f"host.[/yellow]\n  strata-labeller train --job {job_id}"
            )
            raise typer.Exit(0) from None
        except RemoteError as e:
            progress.stop()
            _error(str(e))
            raise typer.Exit(1) from None

    if job.get("state") == "failed":
        _error(f"The round failed on the host: {job.get('error')}")
        raise typer.Exit(1)
    return job["result"]


def _on_another_catalog(where: str, served: dict, catalog, config) -> str:
    """Why a machine on another catalog cannot work with this one, and what to change."""
    if served.get("id"):
        message = (
            f"{where} is on catalog {served.get('name') or '?'} ({served['id']}), "
            f"and this machine is on {catalog.id}."
        )
    else:
        why = served.get("error") or "none — it may predate machines saying which"
        message = f"{where} reports no catalog ({why}), and this machine is on {catalog.id}."
    if not config.url:
        message += (
            f" This catalog's index is SQLite under {config.root}, which only this "
            f"machine can read: the modelling host and the blob server open the index "
            f"themselves, so they can use this catalog only if they run here too. Give "
            f"it a Postgres url to share it, or unset [modelling] url to train here."
        )
    else:
        message += (
            " Make this catalog the default in that machine's config.toml, and restart its service."
        )
    return message


def _unreachable(error: Exception, config) -> str:
    """Why an index could not be opened, in a line, and the likely fix.

    The usual reason on a machine that just switched catalogs: the file
    names the index without its password, and nothing in this shell
    supplies it. See ``docs/adr/0019``.
    """
    import os

    reason = str(error).splitlines()[0] if str(error) else type(error).__name__
    if config.url and "password" in reason.lower() and not os.environ.get("PGPASSWORD"):
        reason += (
            " — $PGPASSWORD is not set. config.toml names the index without its "
            "password, and the environment is where it comes from."
        )
    return reason


def _blob_server_catalog(url: str) -> dict:
    """What a blob server says it serves, from its /healthz."""
    import urllib.request

    with urllib.request.urlopen(f"{url.rstrip('/')}/healthz", timeout=10) as response:
        return json.loads(response.read()).get("catalog") or {}


def _job_state(progress, bar):
    """What to show as a remote job is polled."""

    def show(job: dict) -> None:
        if job.get("state") == "unreachable":
            progress.update(
                bar,
                description=(
                    f"[yellow]Cannot reach the host (attempt "
                    f"{job['attempts']}) — the round is unaffected[/yellow]"
                ),
            )
            return
        stage = job.get("stage", job.get("state", ""))
        if job.get("total"):
            stage = f"{stage} {job['done']:,}/{job['total']:,}"
        progress.update(bar, description=f"Host: {stage}")

    return show
