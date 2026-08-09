"""How Label Studio addresses a sample.

Not what a file *is* — that is a sample type, and it lives in the catalog
because it is a fact about the data rather than about the tool collecting
it. What is left here is Label Studio's wire format: the key its config
reads a task from, and the name that selects a template.

Media is orthogonal to the task: classification applies to an image or a
document alike, while boxes only make sense on images and character spans
only on text. The valid combinations are the template registry, not a
product of the two.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Media:
    #: Used in template names: "<media>_<task>"
    name: str
    #: Key under task["data"] that Label Studio reads the sample from
    data_key: str


IMAGE = Media(name="image", data_key="image")

# Documents are served from the same local-files mount as images; the
# templates ask Label Studio to fetch them with valueType="url", so a text
# project keeps every path, cache and export mechanism images use.
TEXT = Media(name="text", data_key="text")

MEDIA: dict[str, Media] = {m.name: m for m in (IMAGE, TEXT)}
