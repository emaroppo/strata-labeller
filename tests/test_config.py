"""Which catalog a command actually opens.

Settings decide this, and for a long time several commands did not ask
them. The tests here exist because the failure is quiet: a command reads a
real catalog, reports real numbers, and they belong to the wrong index.
"""

from strata.catalog.config import CatalogConfig, Catalogs


def test_a_configured_index_wins_over_a_local_file(tmp_path, monkeypatch):
    """A stale catalog.db must not shadow the index that is configured.

    Five commands opened SQLite whenever the file existed, regardless of
    [catalog] url. After the index moved to Postgres they carried on
    reporting counts from a file nobody was writing to any more, and `train`
    trained on it.
    """
    from strata.catalog import Catalog, LocalBackend
    from strata.labeller.cli._shared import _catalog_if_any
    from strata.labeller.config import Settings

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "catalog"
    Catalog.local(root)
    elsewhere = tmp_path / "elsewhere.db"
    Catalog.create(f"sqlite:///{elsewhere}", LocalBackend(root / "blobs"))

    settings = Settings(
        catalogs=Catalogs(default=CatalogConfig(root=str(root), url=f"sqlite:///{elsewhere}"))
    )
    catalog = _catalog_if_any(settings)
    assert str(elsewhere) in str(catalog.engine.url)


def test_a_local_file_is_used_when_nothing_is_configured(tmp_path, monkeypatch):
    from strata.catalog import Catalog
    from strata.labeller.cli._shared import _catalog_if_any
    from strata.labeller.config import Settings

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "catalog"
    root.mkdir()
    settings = Settings(catalogs=Catalogs(default=CatalogConfig(root=str(root))))
    # Nothing there yet: a project that has never ingested is still a
    # project, so this reports absence rather than failing
    assert _catalog_if_any(settings) is None

    Catalog.local(root)
    assert _catalog_if_any(settings) is not None
