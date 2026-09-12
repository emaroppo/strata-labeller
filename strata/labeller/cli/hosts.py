"""Commands that ask the hosts what they serve."""

from pathlib import Path

import typer
from rich.markup import escape

from ._remote import _blob_server_catalog, _on_another_catalog, _unreachable
from ._shared import (
    CatalogOption,
    ConfigOption,
    _catalog_config,
    _catalog_if_any,
    _error,
    _settings,
    app,
    console,
)


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
    from strata.modelling.client import RemoteError, Trainer

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
