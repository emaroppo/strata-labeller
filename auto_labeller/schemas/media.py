"""What kind of file a project labels.

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
    #: File extensions ``ingest`` picks up, lowercase and without the dot
    extensions: frozenset[str]

    def matches(self, filename: str) -> bool:
        suffix = filename.rsplit(".", 1)
        return len(suffix) == 2 and suffix[1].lower() in self.extensions


IMAGE = Media(
    name="image",
    data_key="image",
    extensions=frozenset({"jpg", "jpeg", "png", "webp", "bmp", "tiff", "gif"}),
)

# Documents are served from the same local-files mount as images; the
# templates ask Label Studio to fetch them with valueType="url", so a text
# project keeps every path, cache and export mechanism images use.
TEXT = Media(
    name="text",
    data_key="text",
    extensions=frozenset({"txt", "md"}),
)

MEDIA: dict[str, Media] = {m.name: m for m in (IMAGE, TEXT)}
