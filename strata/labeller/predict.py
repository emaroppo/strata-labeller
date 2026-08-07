from strata.modelling import Model

from .dataset import Sample
from .project import Project
from .schemas import Prediction


def run_predictions(
    model: Model, samples: list[Sample], project: Project
) -> list[Prediction]:
    """Predict on samples and convert the model's output to storage form.

    The model works with absolute file paths and the schema's own output
    type; everything downstream keys off the data-root-relative sample path
    and Label Studio results.
    """
    schema = project.schema
    paths = [project.sample_file(s.path) for s in samples]
    outputs = model.predict(paths)
    return [
        Prediction(
            path=sample.path,
            results=schema.encode_output(output),
            score=schema.score(output),
            uncertainty=schema.uncertainty(output),
        )
        for sample, output in zip(samples, outputs)
    ]
