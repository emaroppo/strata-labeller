"""How Label Studio addresses a sample: the key a task is read from, and the template name.

Not what a file *is*; that is the catalog's sample type. Media is
orthogonal to the task, and the valid combinations are the template
registry. See ``docs/adr/0013``.
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
