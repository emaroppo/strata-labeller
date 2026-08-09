"""Every label type that ships, driven through the whole contract.

The suite is only worth having if it runs against real types, and these are
also what proves it catches anything: each was written after finding, by
hand, a place that had assumed classification.
"""

import pytest

from strata.labeller.conformance import LabelTypeConformance
from strata.labeller.schemas.bbox import BBoxSchema as LSBBox
from strata.labeller.schemas.classification import ClassificationSchema as LSChoices
from strata.labeller.schemas.span import SpanSchema as LSSpan
from strata.labels import (
    BBoxSchema,
    Box,
    Boxes,
    BoxesPrediction,
    Choices,
    ChoicesPrediction,
    ClassificationSchema,
    Span,
    Spans,
    SpanSchema,
    SpansPrediction,
)

#: Spans and boxes do not round trip through Label Studio yet, and the
#: reason is structural rather than a bug to patch: `strata.labels` and
#: `strata.labeller.schemas` each define their own Box and Span — the second
#: a dataclass predating the first. Encoding works by duck typing on
#: matching attribute names; decoding returns the schema's own type, which
#: the catalog cannot store. `adapter.from_results` then builds a Choices
#: whatever the schema was.
#:
#: strict, so these turn into failures the day the types are unified rather
#: than sitting green and forgotten.
UNUNIFIED = pytest.mark.xfail(
    strict=True,
    reason="labels and the Label Studio schemas define separate Box/Span types",
)


class TestChoices(LabelTypeConformance):
    @pytest.fixture
    def schema(self):
        return ClassificationSchema(classes=["cat", "dog"])

    @pytest.fixture
    def value(self):
        return Choices(values=["cat"])

    @pytest.fixture
    def prediction(self):
        return ChoicesPrediction(values=["cat", "dog"], confidences=[0.9, 0.2])

    @pytest.fixture
    def ls_schema(self):
        return LSChoices(classes=["cat", "dog"])


class TestSpans(LabelTypeConformance):
    @UNUNIFIED
    def test_an_annotation_round_trips_through_label_studio(self, ls_schema, value):
        super().test_an_annotation_round_trips_through_label_studio(ls_schema, value)

    @UNUNIFIED
    def test_a_prediction_encodes_for_label_studio(self, ls_schema, prediction):
        super().test_a_prediction_encodes_for_label_studio(ls_schema, prediction)

    @pytest.fixture
    def schema(self):
        return SpanSchema(classes=["name", "place"])

    @pytest.fixture
    def value(self):
        return Spans(values=[Span(label="name", start=0, end=4)])

    @pytest.fixture
    def prediction(self):
        return SpansPrediction(
            values=[Span(label="name", start=0, end=4)], confidences=[0.8]
        )

    @pytest.fixture
    def ls_schema(self):
        return LSSpan(classes=["name", "place"])


class TestBoxes(LabelTypeConformance):
    @UNUNIFIED
    def test_an_annotation_round_trips_through_label_studio(self, ls_schema, value):
        super().test_an_annotation_round_trips_through_label_studio(ls_schema, value)

    @UNUNIFIED
    def test_a_prediction_encodes_for_label_studio(self, ls_schema, prediction):
        super().test_a_prediction_encodes_for_label_studio(ls_schema, prediction)

    @pytest.fixture
    def schema(self):
        return BBoxSchema(classes=["cat", "dog"])

    @pytest.fixture
    def value(self):
        return Boxes(values=[Box(label="cat", x=0.1, y=0.2, width=0.3, height=0.4)])

    @pytest.fixture
    def prediction(self):
        return BoxesPrediction(
            values=[Box(label="cat", x=0.1, y=0.2, width=0.3, height=0.4)],
            confidences=[0.7],
        )

    @pytest.fixture
    def ls_schema(self):
        return LSBBox(classes=["cat", "dog"])
