"""Talking to the modelling host and the blob server from a command."""

import json
import os
import urllib.request

import typer

from strata.modelling.remote.client import RemoteError, Trainer

from ._shared import _error, _progress, console


def _reattach(host, job_id: str) -> None:
    """Pick up a round that is already running on ``host``, a job id being its own."""
    if not host.url:
        _error("No modelling host configured, so there is no job to reattach to.")
        raise typer.Exit(1)

    trainer = Trainer(host.url, host.token)
    result = _follow(trainer, job_id, host.name)
    run = result["run"]
    _print_run_result(
        run["id"],
        run.get("parent_run_id"),
        result.get("materialised", 0),
        result.get("metrics", {}),
    )


def _trainer(host):
    """A modelling host's client, refused without the token it was started with."""
    if not host.token:
        _error(
            f"No token for the modelling host {host.describe()}. Set "
            f"{host.token_from()} to the same value it was started with."
        )
        raise typer.Exit(1)
    return Trainer(host.url, host.token)


def _print_run_result(
    run_id: str, parent_run_id: str | None, materialised: int, metrics: dict
) -> None:
    """A finished remote round: the run, its parent, and its metrics.

    Takes values, not a record: a fresh round is a ``TrainRecord``, a reattached one a dict.
    """
    console.print(
        f"[green]Run {run_id}[/green]"
        + (f", continuing run {parent_run_id}" if parent_run_id else " (cold)")
    )
    if materialised:
        console.print(f"  [dim]{materialised:,} sample(s) materialised there[/dim]")
    for metric, value in sorted(metrics.items()):
        console.print(f"  {metric}: {value}")


def _follow(trainer, job_id: str, host_name: str = "") -> dict:
    """Watch a round to its end, surviving a network that comes and goes.

    Interrupting this stops watching, not training. See ``docs/adr/0007``.
    """
    # Elapsed, since a long stage looks identical to a stalled one. docs/adr/0031
    with _progress(elapsed=True, transient=True) as progress:
        bar = progress.add_task("Waiting for the host...")

        show = _job_state(progress, bar)

        try:
            job = trainer.follow(job_id, on_state=show)
        except KeyboardInterrupt:
            progress.stop()
            again = f"strata-labeller train --job {job_id}"
            if host_name:
                again += f" --host {host_name}"
            console.print(
                f"[yellow]Stopped watching. The round is still running on the "
                f"host.[/yellow]\n  {again}"
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
            f"it a Postgres url to share it, or train with a [modelling] host whose "
            f"url is empty, which trains here."
        )
    else:
        message += (
            " Make this catalog the default in that machine's config.toml, and restart its service."
        )
    return message


def _unreachable(error: Exception, config) -> str:
    """Why an index could not be opened, in a line, and the likely fix. See ``docs/adr/0019``."""
    reason = str(error).splitlines()[0] if str(error) else type(error).__name__
    if config.url and "password" in reason.lower() and not os.environ.get("PGPASSWORD"):
        reason += (
            " — $PGPASSWORD is not set. config.toml names the index without its "
            "password, and the environment is where it comes from."
        )
    return reason


def _blob_server_catalog(url: str) -> dict:
    """What a blob server says it serves, from its /healthz."""
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
