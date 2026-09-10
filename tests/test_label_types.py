"""Every label type through the Label Studio boundary and the ranking.

Checked against the examples ``strata.labels`` ships, as the catalog and
modelling check their own layers, so a type added there is covered here on
the next upgrade. A type with no Label Studio mapping below fails here —
which is the point: it can be stored and trained on, but not yet reviewed.
"""

import pytest

from strata.labeller.active_learning import certainty, least_confident
from strata.labeller.adapter import prediction_to_results
from strata.labeller.schemas.bbox import BBoxSchema as LSBBox
from strata.labeller.schemas.classification import ClassificationSchema as LSChoices
from strata.labeller.schemas.span import SpanSchema as LSSpan
from strata.labels.examples import EXAMPLES

each_type = pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e.name)

#: How each label type is shown to a reviewer, by the example's name.
LABEL_STUDIO = {
    "choices": lambda: LSChoices(classes=["cat", "dog"]),
    "spans": lambda: LSSpan(classes=["name", "place"]),
    "multi-label spans": lambda: LSSpan(classes=["name", "place"], multi_label=True),
    "overlapping spans": lambda: LSSpan(classes=["name", "place"], overlapping=True),
    "boxes": lambda: LSBBox(classes=["cat", "dog"]),
}


def test_every_label_type_can_be_shown_to_a_reviewer():
    missing = {e.name for e in EXAMPLES} - set(LABEL_STUDIO)
    assert not missing, f"no Label Studio mapping for: {', '.join(sorted(missing))}"


@each_type
def test_an_annotation_round_trips_through_label_studio(example):
    ls_schema = LABEL_STUDIO[example.name]()
    encoded = ls_schema.encode_target(list(example.value.values))
    assert ls_schema.decode_target(encoded) == list(example.value.values)


@each_type
def test_a_prediction_encodes_for_label_studio(example):
    ls_schema = LABEL_STUDIO[example.name]()
    # What a reviewer is shown as a pre-annotation. A type that cannot do
    # this can be labelled but never pre-labelled, which is most of what
    # the loop is for.
    results = prediction_to_results(example.prediction, ls_schema)
    assert ls_schema.decode_target(results) == list(example.prediction.values)


@each_type
def test_the_template_renders(example):
    assert LABEL_STUDIO[example.name]().label_config().strip().startswith("<")


@each_type
def test_uncertainty_reads_the_confidences(example):
    prediction = example.prediction
    # The number beside a task and the order it arrives in have to tell the
    # same story
    assert certainty(prediction) == max(prediction.confidences)
    assert abs(certainty(prediction) - (1.0 - least_confident(prediction))) < 1e-9
