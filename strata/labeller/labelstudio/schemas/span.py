"""Span labelling: labelled character ranges inside a document (NER).

Offsets into the raw text are what matter; the covered substring is kept so a stored
annotation reads on its own. A region carries a list of labels; ``multi_label`` and
``overlapping`` are declared by the label set and default to off. See ``docs/adr/0014``.
"""

from strata.contracts import Span, Spans
from strata.contracts import SpanSchema as StoredSpanSchema

from .base import LabelSchema, Result, strip_volatile
from .media import TEXT, Media
from .render import render_template


class SpanSchema(LabelSchema):
    label_type = "span"
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
        return f"{self.media.name}_{self.label_type}"

    @property
    def data_key(self) -> str:
        return self.media.data_key

    def catalog_schema(self):
        return StoredSpanSchema(
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
            # A whole attribute, so a single-label config stays byte for byte. docs/adr/0029
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
                    # Every label the region carries. docs/adr/0014
                    labels=list(value.get("labels") or []),
                    start=int(value.get("start", 0)),
                    end=int(value.get("end", 0)),
                    text=value.get("text", ""),
                )
            )
        # The order Spans keeps on parse, so a decoded list compares unwrapped. docs/adr/0004
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
