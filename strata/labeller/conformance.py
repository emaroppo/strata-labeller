"""The contract a label type implements, checked end to end.

Subclass it and supply four fixtures; it drives them through every layer
that touches a value between a reviewer answering and a model training.

    class TestBoxes(LabelTypeConformance):
        @pytest.fixture
        def schema(self):
            return BBoxSchema(classes=["cat"])
        ...

This exists because of how the failures look. Six separate places had each
independently assumed classification — a catalog's label set, its stored
annotations, the values it writes into a manifest, the manifest's schema,
the prediction cache, and the wire between hosts. None of them raised. A
value serialises happily whatever it is, and reading it back as the wrong
type parses without complaint and returns something empty: pydantic drops
the fields it does not recognise. So a corpus could be annotated with boxes
and hand back nothing, and the only symptom was a number quietly lower than
it should be.

Every one of those was found by hand, one at a time. This is the thing that
finds the seventh.

Lives in ``labeller`` because it is the only package that may import the
catalog, modelling and the Label Studio boundary together. A type that
passes this is usable for a whole round.
"""

import json

import pytest
from pydantic import TypeAdapter

from strata.catalog import Catalog
from strata.labels import (
    MANIFEST_FORMAT,
    MANIFEST_NAME,
    AnyPrediction,
    AnySchema,
    AnyValue,
    Manifest,
    ManifestSample,
    Prediction,
)
from strata.modelling import PredictionCache

_VALUE = TypeAdapter(AnyValue)
_PREDICTION = TypeAdapter(AnyPrediction)
_SCHEMA = TypeAdapter(AnySchema)


class LabelTypeConformance:
    """Drive a label type through everything between a reviewer and a model."""

    # ------------------------------------------------------------------
    # What an implementer supplies
    # ------------------------------------------------------------------

    @pytest.fixture
    def schema(self):
        """The label schema, as :mod:`strata.labels` describes it."""
        raise NotImplementedError

    @pytest.fixture
    def value(self):
        """An annotation of this type, as a human would have left it."""
        raise NotImplementedError

    @pytest.fixture
    def prediction(self):
        """A model's output of this type, with confidences."""
        raise NotImplementedError

    @pytest.fixture
    def ls_schema(self):
        """The Label Studio mapping for this type."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # The discriminated unions
    # ------------------------------------------------------------------

    def test_a_value_reads_back_as_itself(self, value):
        assert _VALUE.validate_json(value.model_dump_json()) == value

    def test_a_schema_reads_back_as_itself(self, schema):
        assert _SCHEMA.validate_json(schema.model_dump_json()) == schema

    def test_a_prediction_reads_back_as_itself(self, prediction):
        back = _PREDICTION.validate_json(prediction.model_dump_json())
        assert back == prediction
        # The confidences are what a review queue is ordered by. Parsing a
        # prediction as a plain value keeps the answer and loses them, which
        # reorders the queue rather than failing.
        assert back.confidences == prediction.confidences

    def test_a_prediction_is_model_output(self, prediction):
        # So code can ask, rather than infer from which fields are present
        assert isinstance(prediction, Prediction)

    def test_a_prediction_carries_a_confidence_per_thing_asserted(self, prediction):
        assert len(prediction.confidences) == len(prediction.values)

    # ------------------------------------------------------------------
    # The catalog
    # ------------------------------------------------------------------

    @pytest.fixture
    def catalog(self, tmp_path):
        return Catalog.local(tmp_path / "catalog")

    @pytest.fixture
    def media(self) -> str:
        """Which media these samples are, as the catalog records it.

        Defaults to images because that is what every label type was driven
        through before this fixture existed — spans included, whose samples
        are documents rather than pictures. A type exercised as the wrong
        media still passes everything in this suite, which is why it went
        unnoticed; media is read by the serving path, the Label Studio task
        and the template that renders it.
        """
        return "image"

    @pytest.fixture
    def suffix(self, media) -> str:
        """What fixture files are named.

        Cosmetic to the catalog, which records the media it is told rather
        than reading it off a filename. Kept honest anyway: a document
        called ``.bin`` is a misleading thing to leave in a suite whose
        whole job is fidelity.
        """
        return {"text": "txt"}.get(media, "bin")

    @pytest.fixture
    def sample_ids(self, catalog, tmp_path, media, suffix):
        """Two, because a group is indivisible and one cannot be split."""
        sources = []
        for i in range(2):
            source = tmp_path / f"sample{i}.{suffix}"
            source.write_bytes(f"sample {i}".encode())
            sources.append(source)
        return catalog.ingest(sources, media=media)

    @pytest.fixture
    def sample_id(self, sample_ids):
        return sample_ids[0]

    def test_a_label_set_keeps_its_schema(self, catalog, schema):
        catalog.create_label_set("x", schema)
        assert catalog.label_set("x")[1] == schema

    def test_an_annotation_survives_the_catalog(self, catalog, sample_id, schema, value):
        label_set_id = catalog.create_label_set("x", schema)
        catalog.annotate(sample_id, label_set_id, value)
        # The whole point of the catalog: what a human said outlives the
        # tool that collected it, unchanged
        assert catalog.annotation_of(sample_id, label_set_id) == value

    def test_an_empty_answer_is_not_the_same_as_no_answer(
        self, catalog, sample_id, schema, value
    ):
        label_set_id = catalog.create_label_set("x", schema)
        empty = type(value)()
        catalog.annotate(sample_id, label_set_id, empty)
        # A reviewer who looked and found none of the classes present has
        # answered. Only the annotation existing distinguishes that from a
        # sample nobody has seen.
        assert catalog.annotation_of(sample_id, label_set_id) == empty
        assert catalog.labelled(label_set_id, "*")

    # ------------------------------------------------------------------
    # What a model is handed
    # ------------------------------------------------------------------

    def test_a_manifest_carries_the_type(self, tmp_path, schema, value):
        manifest = Manifest(
            format=MANIFEST_FORMAT,
            dataset="d",
            version=1,
            label_set="x",
            label_schema=schema,
            samples=[
                ManifestSample(
                    id=1, checksum="a" * 64, path="files/a", split="train", value=value
                )
            ],
        )
        path = tmp_path / MANIFEST_NAME
        path.write_text(manifest.model_dump_json())

        # A dataset version is the artifact a model trains from, and it has
        # to survive being written to disk and read on another machine
        back = Manifest.model_validate_json(path.read_text())
        assert back.label_schema == schema
        assert back.samples[0].value == value

    def test_a_materialised_dataset_carries_the_type(
        self, catalog, sample_ids, schema, value, tmp_path
    ):
        label_set_id = catalog.create_label_set("x", schema)
        for sample_id in sample_ids:
            catalog.annotate(sample_id, label_set_id, value)
        dataset_id = catalog.create_dataset("d", label_set_id, collections="*")

        directory = catalog.materialise(dataset_id, tmp_path / "out")
        manifest = Manifest.model_validate_json(
            (directory / MANIFEST_NAME).read_text()
        )
        assert manifest.label_schema == schema
        assert manifest.samples[0].value == value

    def test_a_model_is_handed_the_type_it_was_annotated_with(
        self, catalog, sample_ids, schema, value, tmp_path
    ):
        """The last step, and the one that was missing.

        Everything above proves the value survives to a materialised
        dataset. This proves it survives being read back out of one into the
        targets a model trains on, which is a separate piece of code and was
        the only layer between a reviewer and a model that nothing checked.
        It read every value as a classification, so a span or box dataset
        raised on the first sample and no model ever saw one.
        """
        from strata.modelling.handlers import _examples

        label_set_id = catalog.create_label_set("x", schema)
        for sample_id in sample_ids:
            catalog.annotate(sample_id, label_set_id, value)
        dataset_id = catalog.create_dataset("d", label_set_id, collections="*")
        directory = catalog.materialise(dataset_id, tmp_path / "out")
        manifest = json.loads((directory / MANIFEST_NAME).read_text())

        train, val = _examples(directory, manifest)
        assert train or val
        for example in [*train, *val]:
            assert example.target == value

    # ------------------------------------------------------------------
    # Between machines
    # ------------------------------------------------------------------

    def test_a_prediction_survives_the_cache(self, tmp_path, prediction):
        cache = PredictionCache.local(tmp_path / "runs")
        cache.put(1, {"a" * 64: prediction})
        assert cache.get(1, ["a" * 64])["a" * 64] == prediction

    def test_a_prediction_survives_being_tied_to_its_sample(self, prediction, tmp_path):
        """The wrapper a scoring pass returns, on the way to the ranking.

        A prediction is carried out of `predict` paired with the file it was
        made from, and that pairing is the last thing it travels in before
        the review queue is ordered. Typed to one concrete prediction it
        refused every other kind outright, so a scoring pass over a span or
        box project failed after minutes of work rather than at the contract.
        """
        from strata.modelling import ScoredPath

        scored = ScoredPath(path=tmp_path / "a.bin", value=prediction)
        back = ScoredPath.model_validate_json(scored.model_dump_json())
        assert back.value == prediction

    def test_a_prediction_survives_the_wire(self, prediction):
        from strata.modelling.service import PredictionResponse

        response = PredictionResponse(predictions={"a" * 64: prediction})
        back = PredictionResponse.model_validate_json(response.model_dump_json())
        assert back.predictions["a" * 64] == prediction

    # ------------------------------------------------------------------
    # The Label Studio boundary
    # ------------------------------------------------------------------

    def test_an_annotation_round_trips_through_label_studio(self, ls_schema, value):
        encoded = ls_schema.encode_target(list(value.values))
        assert ls_schema.decode_target(encoded) == list(value.values)

    def test_a_prediction_encodes_for_label_studio(self, ls_schema, prediction):
        from .adapter import prediction_to_results

        # What a reviewer is shown as a pre-annotation. A type that cannot
        # do this can be labelled but never pre-labelled, which is most of
        # what the loop is for.
        results = prediction_to_results(prediction, ls_schema)
        assert ls_schema.decode_target(results) == list(prediction.values)

    def test_the_template_renders(self, ls_schema):
        config = ls_schema.label_config()
        assert config.strip().startswith("<")

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def test_uncertainty_reads_the_confidences(self, prediction):
        from .active_learning import certainty, least_confident

        # The number beside a task and the order it arrives in have to tell
        # the same story
        assert certainty(prediction) == max(prediction.confidences)
        assert abs(certainty(prediction) - (1.0 - least_confident(prediction))) < 1e-9
