"""Classification: one or more classes for a whole sample."""

from strata.labels import Choices

from .base import LabelSchema, Result, strip_volatile
from .media import IMAGE, Media
from .render import render_template


class ClassificationSchema(LabelSchema):
    value_type = Choices
    """Classes for a whole sample, whatever the sample is made of."""

    task = "classification"
    control_tag = "Choices"

    def __init__(
        self,
        classes: list[str],
        media: Media = IMAGE,
        choice: str = "multiple",
        from_name: str = "label",
        to_name: str | None = None,
    ):
        self.classes = list(classes)
        self.media = media
        self.choice = choice
        self.from_name = from_name
        self.to_name = to_name or media.data_key

    @property
    def type(self) -> str:
        return f"{self.media.name}_{self.task}"

    @property
    def data_key(self) -> str:
        return self.media.data_key

    # ------------------------------------------------------------------
    # Label Studio config
    # ------------------------------------------------------------------

    def catalog_schema(self):
        from strata.labels import ClassificationSchema as Stored

        return Stored(classes=list(self.classes), multiple=self.choice != "single")

    def label_config(self) -> str:
        return render_template(
            self.type,
            classes=self.classes,
            label_tag="Choice",
            from_name=self.from_name,
            to_name=self.to_name,
            choice=self.choice,
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def canonicalize(self, results: list[dict]) -> list[Result]:
        return [strip_volatile(r) for r in results if r.get("type") == "choices"]

    def decode_target(self, results: list[Result]) -> list[str]:
        return [choice for r in results for choice in r.get("value", {}).get("choices", [])]

    def encode_target(self, target: list[str]) -> list[Result]:
        if not target:
            return []
        return [
            {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "choices",
                "value": {"choices": list(target)},
            }
        ]
