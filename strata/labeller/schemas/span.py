"""Span labelling: labelled character ranges inside a document (NER).

Label Studio addresses spans by character offsets into the raw text, and
carries the covered substring alongside. Offsets are what matter; the text
is kept because it makes a stored annotation readable on its own.
"""

from strata.labels import Span, Spans

from .base import LabelSchema, Result, _confidences, strip_volatile
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
            indent="    ",
            from_name=self.from_name,
            to_name=self.to_name,
        )

    def canonicalize(self, results: list[dict]) -> list[Result]:
        return [strip_volatile(r) for r in results if r.get("type") == "labels"]

    def decode_target(self, results: list[Result]) -> list[Span]:
        spans: list[Span] = []
        for r in results:
            value = r.get("value", {})
            labels = value.get("labels") or []
            spans.append(
                Span(
                    label=labels[0] if labels else "",
                    start=int(value.get("start", 0)),
                    end=int(value.get("end", 0)),
                    text=value.get("text", ""),
                )
            )
        # Reading order makes stored annotations and model targets comparable
        return sorted(spans, key=lambda s: (s.start, s.end))

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
                    "labels": [span.label],
                },
            }
            for span in target
        ]

    def encode_output(self, output) -> list[Result]:
        return self.encode_target(list(output.values))

    def score(self, output) -> float:
        # As trustworthy as its least certain span; claiming nothing scores zero
        return min(_confidences(output), default=0.0)

    def uncertainty(self, output) -> float:
        """Spans near the decision threshold are the informative ones.

        A document the model found nothing in is maximally uncertain: it
        either contains nothing or the model missed everything, and only a
        reader settles which.
        """
        scores = _confidences(output)
        if not scores:
            return 1.0
        return max(1.0 - abs(s - 0.5) * 2.0 for s in scores)

    def classes_in_use(self, results_lists: list[list[Result]]) -> list[str]:
        seen: set[str] = set()
        for results in results_lists:
            for span in self.decode_target(results):
                if span.label:
                    seen.add(span.label)
        return sorted(seen)
