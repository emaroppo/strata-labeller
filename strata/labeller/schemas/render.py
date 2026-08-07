"""Rendering the packaged label config templates."""

from importlib import resources

from .base import TemplateSyntax

TEMPLATE_PACKAGE = "strata.labeller.label_configs"


def read_template(name: str) -> str:
    return (
        resources.files(TEMPLATE_PACKAGE).joinpath(f"{name}.xml").read_text(encoding="utf-8")
    )


def render_template(
    name: str,
    classes: list[str],
    label_tag: str,
    indent: str = "      ",
    **params: str,
) -> str:
    """Fill a template's @placeholders, rendering the class list itself.

    Hotkeys are assigned in order so review can stay on the keyboard; Label
    Studio only binds single digits, so classes past the ninth go without.
    """
    lines = []
    for i, value in enumerate(classes, start=1):
        hotkey = f' hotkey="{i}"' if i <= 9 else ""
        lines.append(f'{indent}<{label_tag} value="{value}"{hotkey}/>')
    template = TemplateSyntax(read_template(name))
    return template.substitute(labels="\n".join(lines), **params).rstrip("\n")
