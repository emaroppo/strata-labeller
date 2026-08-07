"""Bounding boxes: zero or more labelled rectangles per image.

Label Studio stores box geometry as percentages of the image (0-100) plus
the original pixel dimensions. Models work in fractions of the image
(0-1) with a top-left origin, so the conversion lives here and nowhere
else.
"""

from dataclasses import dataclass, field

from .base import LabelSchema, Result, strip_volatile
from .media import IMAGE, Media
from .render import render_template

# Kept alongside coordinates: percentages are meaningless without them
GEOMETRY_FIELDS = ("original_width", "original_height", "image_rotation")


@dataclass
class Box:
    label: str
    # Fractions of the image, top-left origin
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0
    score: float = 1.0


@dataclass
class BoxOutput:
    boxes: list[Box] = field(default_factory=list)


class BBoxSchema(LabelSchema):
    task = "bbox"
    control_tag = "RectangleLabels"

    def __init__(
        self,
        classes: list[str],
        media: Media = IMAGE,
        from_name: str = "label",
        to_name: str | None = None,
    ):
        self.classes = list(classes)
        self.media = media
        self.from_name = from_name
        self.to_name = to_name or media.data_key

    @property
    def type(self) -> str:
        return f"{self.media.name}_{self.task}"

    @property
    def data_key(self) -> str:
        return self.media.data_key

    def label_config(self) -> str:
        return render_template(
            self.type,
            classes=self.classes,
            label_tag="Label",
            from_name=self.from_name,
            to_name=self.to_name,
        )

    def canonicalize(self, results: list[dict]) -> list[Result]:
        return [
            strip_volatile(r, keep=GEOMETRY_FIELDS)
            for r in results
            if r.get("type") == "rectanglelabels"
        ]

    def decode_target(self, results: list[Result]) -> list[Box]:
        boxes: list[Box] = []
        for r in results:
            value = r.get("value", {})
            labels = value.get("rectanglelabels") or []
            boxes.append(
                Box(
                    label=labels[0] if labels else "",
                    x=value.get("x", 0.0) / 100.0,
                    y=value.get("y", 0.0) / 100.0,
                    width=value.get("width", 0.0) / 100.0,
                    height=value.get("height", 0.0) / 100.0,
                    rotation=value.get("rotation", 0.0),
                )
            )
        return boxes

    def encode_target(self, target: list[Box]) -> list[Result]:
        return [
            {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "rectanglelabels",
                "value": {
                    "x": box.x * 100.0,
                    "y": box.y * 100.0,
                    "width": box.width * 100.0,
                    "height": box.height * 100.0,
                    "rotation": box.rotation,
                    "rectanglelabels": [box.label],
                },
            }
            for box in target
        ]

    def encode_output(self, output: BoxOutput) -> list[Result]:
        return self.encode_target(output.boxes)

    def score(self, output: BoxOutput) -> float:
        # A detection is only as trustworthy as its weakest box; an empty
        # prediction claims nothing, so it scores zero
        return min((b.score for b in output.boxes), default=0.0)

    def uncertainty(self, output: BoxOutput) -> float:
        """Boxes sitting near the decision threshold are the informative ones.

        An image with no boxes at all is maximally uncertain: either the
        model found nothing, or it missed everything, and only a human
        settles which.
        """
        if not output.boxes:
            return 1.0
        return max(1.0 - abs(b.score - 0.5) * 2.0 for b in output.boxes)

    def classes_in_use(self, results_lists: list[list[Result]]) -> list[str]:
        seen: set[str] = set()
        for results in results_lists:
            for box in self.decode_target(results):
                if box.label:
                    seen.add(box.label)
        return sorted(seen)
