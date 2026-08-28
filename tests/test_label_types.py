"""Every label type that ships, driven through the whole contract.

The suite is only worth having if it runs against real types, and these are
also what proves it catches anything: each was written after finding, by
hand, a place that had assumed classification.
"""

import pytest

from strata.catalog import Catalog
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
    SchemaError,
    Span,
    Spans,
    SpanSchema,
    SpansPrediction,
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
    @pytest.fixture
    def media(self):
        # Spans are character ranges in a document, so the samples carrying
        # them are text. Driven through the catalog as images until now.
        return "text"

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


class TestMultiLabelSpans(LabelTypeConformance):
    """A region carrying two labels, all the way through.

    Label Studio has always been able to express this; the layer storing it
    could not, and dropped the second label without a word. Driving it
    through the whole contract is what proves the two now agree.
    """

    @pytest.fixture
    def media(self):
        return "text"

    @pytest.fixture
    def schema(self):
        return SpanSchema(classes=["name", "place"], multi_label=True)

    @pytest.fixture
    def value(self):
        return Spans(values=[Span(labels=["name", "place"], start=0, end=4)])

    @pytest.fixture
    def prediction(self):
        return SpansPrediction(
            values=[Span(labels=["name", "place"], start=0, end=4)], confidences=[0.8]
        )

    @pytest.fixture
    def ls_schema(self):
        return LSSpan(classes=["name", "place"], multi_label=True)


class TestOverlappingSpans(LabelTypeConformance):
    """Two regions that intersect, where the label set says they may."""

    @pytest.fixture
    def media(self):
        return "text"

    @pytest.fixture
    def schema(self):
        return SpanSchema(classes=["name", "place"], overlapping=True)

    @pytest.fixture
    def value(self):
        return Spans(
            values=[
                Span(labels=["name"], start=0, end=8),
                Span(labels=["place"], start=4, end=12),
            ]
        )

    @pytest.fixture
    def prediction(self):
        return SpansPrediction(
            values=[
                Span(labels=["name"], start=0, end=8),
                Span(labels=["place"], start=4, end=12),
            ],
            confidences=[0.8, 0.4],
        )

    @pytest.fixture
    def ls_schema(self):
        return LSSpan(classes=["name", "place"], overlapping=True)


def test_an_undeclared_overlap_is_refused_where_it_would_be_stored(tmp_path):
    """The refusal has to fire on the path a reviewer's answer takes.

    Label Studio will let anyone draw two regions across one phrase. Until
    the label set says whether that is meaningful here, storing it means a
    tagger silently training on whichever of the two came last.
    """
    catalog = Catalog.local(tmp_path / "catalog")
    document = tmp_path / "doc.txt"
    document.write_text("Ada Lovelace worked here")
    [sample_id] = catalog.ingest([document], media="text")
    label_set_id = catalog.create_label_set(
        "x", SpanSchema(classes=["name", "place"])
    )

    with pytest.raises(SchemaError, match="overlap"):
        catalog.annotate(
            sample_id,
            label_set_id,
            Spans(
                values=[
                    Span(labels=["name"], start=0, end=12),
                    Span(labels=["place"], start=4, end=20),
                ]
            ),
        )


class TestBoxes(LabelTypeConformance):
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
