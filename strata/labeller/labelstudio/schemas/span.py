"""Span labelling: labelled character ranges inside a document (NER).

Label Studio addresses spans by character offsets into the raw text, and
carries the covered substring alongside. Offsets are what matter; the text
is kept because it makes a stored annotation readable on its own.

Its model is a region with a *list* of labels, and this read the first and
dropped the rest — an annotation tool being more permissive than the layer
storing what it produced. What a label set allows is now declared:
``multi_label`` puts ``choice="multiple"`` on the control so a reviewer can
say it, and ``overlapping`` says two regions may intersect. Both default to
off, which is what every existing project means.
"""

from strata.labels import Span, Spans

from .base import LabelSchema, Result, strip_volatile
from .media import TEXT, Media
from .render import render_template


class SpanSchema(LabelSchema):
    task = "span"
    value_type = Spans
    control_tag = "Labels"

    def __init__(
        self,
        classes: list[str],
        media: Media = TEXT,
        from_name: str = "label",
        to_name: str | None = None,
        multi_label: bool = False,
        overlapping: bool = False,
    ):
        self.classes = list(classes)
        self.media = media
        self.from_name = from_name
        self.to_name = to_name or media.data_key
        self.multi_label = bool(multi_label)
        self.overlapping = bool(overlapping)

    @property
    def type(self) -> str:
        return f"{self.media.name}_{self.task}"

    @property
    def data_key(self) -> str:
        return self.media.data_key

    def catalog_schema(self):
        from strata.labels import SpanSchema as Stored

        return Stored(
            classes=list(self.classes),
            multi_label=self.multi_label,
            overlapping=self.overlapping,
        )

    def label_config(self) -> str:
        return render_template(
            self.type,
            classes=self.classes,
            label_tag="Label",
            indent="    ",
            from_name=self.from_name,
            to_name=self.to_name,
            # Rendered as a whole attribute rather than a value, so a
            # single-label project's config is byte for byte what it was.
            # A config that changes shape re-validates in Label Studio and
            # is one more thing to explain in a diff.
            choice=' choice="multiple"' if self.multi_label else "",
        )

    def canonicalize(self, results: list[dict]) -> list[Result]:
        return [strip_volatile(r) for r in results if r.get("type") == "labels"]

    def decode_target(self, results: list[Result]) -> list[Span]:
        spans: list[Span] = []
        for r in results:
            value = r.get("value", {})
            spans.append(
                Span(
                    # Every label the region carries. Reading the first was
                    # a silent loss: nothing raised, and the second label a
                    # reviewer chose simply never reached the catalog.
                    labels=list(value.get("labels") or []),
                    start=int(value.get("start", 0)),
                    end=int(value.get("end", 0)),
                    text=value.get("text", ""),
                )
            )
        # The order Spans keeps on parse (docs/adr/0004), so a decoded list
        # compares before it is wrapped
        return sorted(spans, key=lambda s: (s.start, s.end, tuple(s.labels)))

    def encode_target(self, target: list[Span]) -> list[Result]:
        return [
            {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "labels",
                "value": {
                    "start": span.start,
                    "end": span.end,
                    "text": span.text,
                    "labels": list(span.labels),
                },
            }
            for span in target
        ]
