"""The dependency graph, enforced rather than described.

Phase 0 said an import violating the graph should fail the build. It never
did, and the graph has held by discipline — which works until the day
someone reaches for the obvious import and nothing objects.

The rules are not stylistic. The catalog is the durable asset: annotations
outlive the tool that collected them, so it must not import the labelling
tool or an ML framework, or a five-year-old corpus needs both installed to
be read. The training core takes a directory and a manifest, so a model can
be tested against a fixture and run where no database exists — the service
layer is the one place allowed to know a catalog.

Lives here because `labeller` is the only package permitted to import the
others, so it is the natural place to assert what they may not.
"""

import ast
import pathlib

import pytest

#: Which strata packages each may import.
ALLOWED = {
    "labels": set(),
    "common": set(),
    "catalog": {"labels", "common"},
    "modelling": {"labels", "catalog", "common"},
    "labeller": {"labels", "catalog", "modelling"},
}

#: Third-party imports that would undo the point of a package.
FORBIDDEN = {
    "labels": {"torch", "timm", "transformers", "sqlalchemy", "boto3",
               "fastapi", "label_studio_sdk"},
    # Thin by design: nothing but what its extras name, and never the
    # frameworks or the value types, or every consumer would carry them.
    "common": {"torch", "timm", "transformers", "boto3", "fastapi",
               "label_studio_sdk", "pydantic"},
    "catalog": {"torch", "timm", "transformers", "label_studio_sdk"},
    "modelling": {"label_studio_sdk"},
    "labeller": set(),
}

#: `modelling` may import `catalog`, but only from its service layer.
CATALOG_IN_MODELLING = {"service.py"}


def _imports(path: pathlib.Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def _modules(package: str):
    """Every module in a package, and never an empty list.

    A path that resolves to nothing makes all of these pass without reading
    anything, which is how this suite spent its first minutes: green, and
    scanning a directory that does not exist.
    """
    root = pathlib.Path(__file__).resolve().parents[2] / package / "strata" / package
    found = sorted(root.rglob("*.py"))
    assert found, f"no modules found under {root} — this suite would pass vacuously"
    return found


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_a_package_imports_only_what_it_may(package):
    offences = []
    for path in _modules(package):
        for name in _imports(path):
            if not name.startswith("strata."):
                continue
            other = name.split(".")[1]
            if other != package and other not in ALLOWED[package]:
                offences.append(f"{path.name} imports strata.{other}")
    assert not offences, f"{package}: " + "; ".join(offences)


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_a_package_avoids_what_would_undo_it(package):
    offences = []
    for path in _modules(package):
        for name in _imports(path):
            top = name.split(".")[0]
            if top in FORBIDDEN[package]:
                offences.append(f"{path.name} imports {top}")
    assert not offences, f"{package}: " + "; ".join(offences)


def test_only_the_service_layer_of_modelling_knows_a_catalog():
    """The training core takes a directory and a manifest, and nothing else.

    That is what makes a dataset version portable and the core testable
    against a fixture. A catalog import anywhere else undoes both, quietly.
    """
    offenders = {
        path.name
        for path in _modules("modelling")
        if any(name.startswith("strata.catalog") for name in _imports(path))
    }
    assert offenders <= CATALOG_IN_MODELLING, (
        f"catalog imported outside the service layer: "
        f"{', '.join(sorted(offenders - CATALOG_IN_MODELLING))}"
    )
