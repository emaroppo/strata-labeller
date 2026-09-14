"""The schema layer: what a task type stores, and how it converts.

Everything else keys off these conversions, so the round trips are the load
-bearing tests: a stored annotation has to reach a model unchanged, and a
model's output has to reach Label Studio in the shape it expects.
"""

import pytest

from strata.labeller.labelstudio import schemas
from strata.labeller.labelstudio.schemas import (
    BBoxSchema,
    ClassificationSchema,
    SpanSchema,
    strip_volatile,
)
from strata.labeller.labelstudio.schemas.media import IMAGE, TEXT
from strata.labels import (
    Box,
    Span,
)

# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------


def test_classification_target_round_trip():
    schema = ClassificationSchema(["cat", "dog", "bird"])
    assert schema.decode_target(schema.encode_target(["cat", "bird"])) == ["cat", "bird"]


def test_classification_encodes_nothing_for_an_empty_target():
    # An empty answer is stored as no results at all; dataset.annotated is
    # what records that a human looked
    assert ClassificationSchema(["cat"]).encode_target([]) == []


def test_classification_canonicalize_drops_volatile_and_foreign_results():
    schema = ClassificationSchema(["cat"])
    results = schema.canonicalize(
        [
            {
                "id": "abc123",
                "lead_time": 4.2,
                "created_at": "2026-01-01",
                "origin": "manual",
                "from_name": "label",
                "to_name": "image",
                "type": "choices",
                "value": {"choices": ["cat"]},
            },
            # A second control in the same config: not this schema's business
            {"from_name": "notes", "type": "textarea", "value": {"text": ["hi"]}},
        ]
    )
    assert results == [
        {
            "from_name": "label",
            "to_name": "image",
            "type": "choices",
            "value": {"choices": ["cat"]},
        }
    ]


def test_media_decides_the_type_and_data_key():
    assert ClassificationSchema(["a"], media=IMAGE).type == "image_classification"
    assert ClassificationSchema(["a"], media=TEXT).type == "text_classification"
    assert ClassificationSchema(["a"], media=TEXT).data_key == "text"
    # to_name defaults to the media's data key so the config wires up
    assert ClassificationSchema(["a"], media=TEXT).to_name == "text"


# ----------------------------------------------------------------------
# Bounding boxes
# ----------------------------------------------------------------------


def test_bbox_converts_between_fractions_and_percentages():
    schema = BBoxSchema(["cat"])
    encoded = schema.encode_target([Box(label="cat", x=0.1, y=0.25, width=0.5, height=0.2)])
    value = encoded[0]["value"]
    # Label Studio speaks percentages of the image; models speak fractions
    assert (value["x"], value["y"], value["width"], value["height"]) == (10.0, 25.0, 50.0, 20.0)
    assert value["rectanglelabels"] == ["cat"]

    box = schema.decode_target(encoded)[0]
    assert (box.x, box.y, box.width, box.height) == pytest.approx((0.1, 0.25, 0.5, 0.2))
    assert box.label == "cat"


def test_bbox_canonicalize_keeps_the_geometry_fields():
    schema = BBoxSchema(["cat"])
    results = schema.canonicalize(
        [
            {
                "id": "drop-me",
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "original_width": 1920,
                "original_height": 1080,
                "image_rotation": 0,
                "value": {
                    "x": 1.0,
                    "y": 2.0,
                    "width": 3.0,
                    "height": 4.0,
                    "rectanglelabels": ["cat"],
                },
            }
        ]
    )
    # Percentages mean nothing without the dimensions they are percentages of
    assert results[0]["original_width"] == 1920
    assert results[0]["original_height"] == 1080
    assert "id" not in results[0]


# ----------------------------------------------------------------------
# Spans
# ----------------------------------------------------------------------


def test_span_offsets_recover_the_source_text():
    document = "Ada met Bob in Rome"
    schema = SpanSchema(["PERSON", "PLACE"])
    spans = [
        Span(labels=["PERSON"], start=0, end=3, text="Ada"),
        Span(labels=["PLACE"], start=15, end=19, text="Rome"),
    ]
    decoded = schema.decode_target(schema.encode_target(spans))
    for span in decoded:
        # The offsets are what matter; the stored text must agree with them
        assert document[span.start : span.end] == span.text


def test_span_decode_sorts_into_reading_order():
    schema = SpanSchema(["X"])
    encoded = schema.encode_target(
        [
            Span(labels=["X"], start=10, end=12, text="th"),
            Span(labels=["X"], start=0, end=3, text="Ada"),
            Span(labels=["X"], start=10, end=11, text="t"),
        ]
    )
    decoded = schema.decode_target(encoded)
    assert [(s.start, s.end) for s in decoded] == [(0, 3), (10, 11), (10, 12)]


# ----------------------------------------------------------------------
# strip_volatile
# ----------------------------------------------------------------------


def test_strip_volatile_drops_churn_and_unknown_keys():
    cleaned = strip_volatile(
        {
            "id": "x",
            "origin": "manual",
            "lead_time": 1.0,
            "parent_id": None,
            "from_name": "label",
            "to_name": "image",
            "type": "choices",
            "value": {"choices": ["cat"]},
            "something_else": 1,
        }
    )
    assert set(cleaned) == {"from_name", "to_name", "type", "value"}


def test_strip_volatile_keeps_what_it_is_told_to():
    cleaned = strip_volatile({"type": "x", "original_width": 10}, keep=("original_width",))
    assert cleaned["original_width"] == 10


# ----------------------------------------------------------------------
# The template registry
# ----------------------------------------------------------------------


def test_templates_cover_the_valid_combinations_and_not_their_product():
    available = schemas.available_templates()
    assert "image_bbox" in available
    assert "text_span" in available
    # The matrix has holes on purpose: boxes only on images, spans only on text
    assert "text_bbox" not in available
    assert "image_span" not in available


def test_from_template_rejects_an_unknown_template():
    with pytest.raises(schemas.SchemaError, match="Unknown template"):
        schemas.from_template("image_segmentation", ["a"])


def test_from_template_rejects_a_parameter_the_template_does_not_take():
    with pytest.raises(schemas.SchemaError, match="takes no parameter"):
        schemas.from_template("image_bbox", ["a"], choice="single")


def test_from_template_passes_parameters_through():
    schema = schemas.from_template("image_classification", ["a"], choice="single")
    assert schema.choice == "single"


def test_from_label_config_reads_control_names_classes_and_media():
    xml = """
    <View>
      <Labels name="ner" toName="doc">
        <Label value="PERSON"/>
        <Label value="PLACE"/>
      </Labels>
      <Text name="doc" value="$text" valueType="url"/>
    </View>
    """
    schema = schemas.from_label_config(xml)
    # The XML is authoritative: it is what annotations will reference
    assert isinstance(schema, SpanSchema)
    assert schema.classes == ["PERSON", "PLACE"]
    assert schema.from_name == "ner"
    assert schema.to_name == "doc"
    assert schema.type == "text_span"


def test_from_label_config_rejects_an_unsupported_control():
    xml = '<View><Image name="image" value="$image"/><TextArea name="t" toName="image"/></View>'
    with pytest.raises(schemas.SchemaError, match="No supported labeling control"):
        schemas.from_label_config(xml)


def test_from_label_config_rejects_a_config_with_no_media_tag():
    xml = '<View><Choices name="l" toName="x"><Choice value="a"/></Choices></View>'
    with pytest.raises(schemas.SchemaError, match="No supported media tag"):
        schemas.from_label_config(xml)


def test_generated_config_contains_the_classes_and_wires_the_controls():
    schema = ClassificationSchema(["cat", "dog"], media=IMAGE, choice="single")
    xml = schema.label_config()
    assert '<Choice value="cat" hotkey="1"/>' in xml
    assert '<Choice value="dog" hotkey="2"/>' in xml
    assert 'choice="single"' in xml
    # A generated config must be readable back into the same schema
    assert schemas.from_label_config(xml).classes == ["cat", "dog"]


def test_generated_config_stops_assigning_hotkeys_past_the_ninth():
    schema = ClassificationSchema([f"c{i}" for i in range(1, 12)])
    xml = schema.label_config()
    assert '<Choice value="c9" hotkey="9"/>' in xml
    # Label Studio only binds single digits
    assert '<Choice value="c10"/>' in xml
