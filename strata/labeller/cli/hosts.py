"""Commands that ask the hosts what they serve."""

from pathlib import Path

import typer
from rich.markup import escape

from ._remote import _blob_server_catalog, _on_another_catalog, _unreachable
from ._shared import (
    CatalogOption,
    ConfigOption,
    ProjectOption,
    _catalog_config,
    _catalog_if_any,
    _error,
    _settings,
    app,
    console,
)


@app.command()
def models(
    project_path: Path | None = ProjectOption,
    config_path: Path = ConfigOption,
    as_json: bool = typer.Option(False, "--json", help="Emit the list as JSON"),
) -> None:
    """What the machine that trains can serve, asked before the first round is.

    With a [modelling] url in config.toml this is the host's list, read
    from what it has installed; without one it is this machine's, since
    rounds then run here. The backend is the only authority on what it can
    serve, so the list is not used to validate a round — the round is
    checked when it arrives — but nothing else asks until a train, which
    can be after labelling a few hundred samples. With a project, says
    whether its model is on the list: a file reference runs here and never
    on a host.
    """
    import json

    from strata.modelling.plugins.registry import available
    from strata.modelling.remote.client import RemoteError, Trainer

    from ..project import LabellingProject, ProjectError

    settings = _settings(config_path)
    if settings.modelling.url:
        where, url = "modelling host", settings.modelling.url
        try:
            served = Trainer(url, settings.modelling.token).models()
        except (RemoteError, OSError, ValueError) as e:
            _error(f"The modelling host at {url} did not answer: {escape(str(e))}")
            raise typer.Exit(1) from None
    else:
        where, url = "this machine", ""
        served = available()

    try:
        project = LabellingProject.load(project_path)
    except ProjectError:
        if project_path is not None:
            raise
        project = None

    verdict = None
    ref, servable = "", False
    if project is not None:
        ref = project.model.ref
        if ":" in ref:
            verdict = (
                "a file reference, which runs here and not on a host"
                if url
                else "a file reference, which runs here"
            )
            servable = not url
        else:
            servable = ref in served
            verdict = "served" if servable else f"not among what the {where} serves"

    if as_json:
        payload = {"where": where, "url": url, "models": served}
        if project is not None:
            payload["project"] = {"name": project.name, "model": ref, "servable": servable}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        console.print(f"[bold]{where}[/bold]  {escape(url)}".rstrip())
        if not served:
            console.print("  [yellow]nothing registered under strata.models[/yellow]")
        for name, target in sorted(served.items()):
            console.print(f"  {name:<16} {escape(target)}")
        if project is not None:
            colour = "green" if servable else "red"
            console.print(
                f"[bold]project {escape(project.name)}[/bold]  model {escape(ref)!s} — "
                f"[{colour}]{verdict}[/{colour}]"
            )
    if project is not None and not servable:
        raise typer.Exit(1)


@app.command(name="catalog-check")
def catalog_check(
    catalog_name: str = CatalogOption,
    config_path: Path = ConfigOption,
) -> None:
    """Ask the blob server and the modelling host which catalog they are on.

    Each machine reads its own config.toml, so pointing the setup at another
    catalog is an edit on each, and a machine missed is what this catches: a
    blob server left on the old catalog answers 404 for every new sample, and
    a modelling host left on it refuses every round.
    """
    from strata.modelling.remote.client import RemoteError, Trainer

    settings = _settings(config_path)
    config = _catalog_config(settings, catalog_name)
    try:
        catalog = _catalog_if_any(settings, catalog_name)
    except Exception as e:  # the answer to "which catalog", not a crash
        _error(f"Cannot open this machine's catalog: {_unreachable(e, config)}")
        raise typer.Exit(1) from None
    if catalog is None:
        _error(f"No catalog at {config.root} yet, so there is nothing to compare against.")
        raise typer.Exit(1)
    name = catalog_name or settings.catalogs.default_name or "default"
    console.print(f"[bold]this machine[/bold]  {escape(name)}  {catalog.id}")

    hosts = []
    if config.serve_url:
        hosts.append(
            ("blob server", config.serve_url, lambda: _blob_server_catalog(config.serve_url))
        )
    if settings.modelling.url:
        trainer = Trainer(settings.modelling.url, settings.modelling.token)
        hosts.append(("modelling host", settings.modelling.url, trainer.served_catalog))
    if not hosts:
        console.print(
            escape("No serve_url and no [modelling] url, so no other machine reads a catalog.")
        )
        return

    ok = True
    for label, url, ask in hosts:
        try:
            served = ask()
        except (RemoteError, OSError, ValueError) as e:
            ok = False
            console.print(f"[bold]{label}[/bold]  {url}  [red]unreachable[/red] — {escape(str(e))}")
            continue
        if served.get("id") == catalog.id:
            console.print(f"[bold]{label}[/bold]  {url}  [green]same catalog[/green]")
        else:
            ok = False
            console.print(f"[bold]{label}[/bold]  {url}  [red]another catalog[/red]")
            _error(_on_another_catalog(f"The {label} at {url}", served, catalog, config))
    raise typer.Exit(0 if ok else 1)
