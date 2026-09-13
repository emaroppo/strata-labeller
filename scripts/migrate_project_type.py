"""Rewrite a project written before sample types.

    uv run python packages/labeller/scripts/migrate_project_type.py projects/my-project

``[data] kind`` said two things at once: what a sample was, and how samples
grouped. Only one of them had a name — ``images`` or ``frames`` — so a
satellite scene, which is an image that groups by scene and carries
coordinates, had nowhere to go.

``[data] type`` names a registered sample type instead, and grouping is that
type's business. The mapping is:

    kind = "frames"   ->  type = "frames"
    kind = "images"   ->  type = the template's media, image or text

Which is why this cannot be a default: for anything but frames, the answer
is in the label config rather than in the setting being replaced.
"""

import re
import sys
import tomllib
from pathlib import Path

MEDIA_BY_TEMPLATE = {
    "image_classification": "image",
    "image_bbox": "image",
    "text_classification": "text",
    "text_span": "text",
}


def migrate(project_root: Path) -> str:
    toml_path = project_root / "project.toml"
    if not toml_path.exists():
        raise SystemExit(f"No project.toml under {project_root}")

    data = tomllib.loads(toml_path.read_text())
    if data.get("data", {}).get("type"):
        raise SystemExit(f"{toml_path} already names a type; nothing to do.")

    kind = data.get("data", {}).get("kind", "images")
    if kind == "frames":
        sample_type = "frames"
    else:
        template = data.get("label_config", {}).get("template", "image_classification")
        sample_type = MEDIA_BY_TEMPLATE.get(template)
        if sample_type is None:
            raise SystemExit(
                f"Cannot tell what {toml_path} labels: template {template!r} is "
                f"not one this knows. Set [data] type by hand — "
                f"'strata-labeller types' lists what is installed."
            )

    text = toml_path.read_text()
    line = f'type = "{sample_type}"'
    if re.search(r"^\s*kind\s*=.*$", text, flags=re.MULTILINE):
        # In place, so a hand-written project.toml keeps its comments and
        # the setting stays where its author put it
        text = re.sub(r"^\s*kind\s*=.*$", line, text, count=1, flags=re.MULTILINE)
    elif re.search(r"^\[data\]\s*$", text, flags=re.MULTILINE):
        text = re.sub(r"^(\[data\]\s*)$", rf"\1\n{line}", text, count=1, flags=re.MULTILINE)
    else:
        text = text.rstrip("\n") + f"\n\n[data]\n{line}\n"
    toml_path.write_text(text)
    return sample_type


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1])
    chosen = migrate(root)
    print(f"{root / 'project.toml'}: [data] type = {chosen!r}")
    if chosen == "frames":
        print("Grouping is unchanged: frames still group by the folder they sit in.")
