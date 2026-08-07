"""FastAPI ML backend serving live predictions to Label Studio.

The project to serve comes from the AUTO_LABELLER_PROJECT environment
variable (set by ``auto-labeller serve --project``), since uvicorn imports
this module rather than calling into it.
"""

from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI
from pydantic import Model as PydanticModel

from strata.modelling import Model

from .project import Project

app = FastAPI(title="auto-labeller ML backend")
_model: Model | None = None
_project: Project | None = None


class PredictRequest(PydanticModel):
    tasks: list[dict]
    project: str | None = None


@app.on_event("startup")
def startup() -> None:
    global _model, _project
    _project = Project.load()
    _model = _project.load_model()
    checkpoint = _project.latest_checkpoint()
    if checkpoint:
        _model.load(checkpoint)


@app.get("/health")
def health() -> dict:
    return {"status": "UP"}


@app.post("/setup")
def setup() -> dict:
    return {"model_version": "latest"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    assert _model is not None and _project is not None
    prefix = _project.label_studio.local_files_prefix
    data_key = _project.schema.data_key

    paths: list[Path] = []
    for task in request.tasks:
        url = task.get("data", {}).get(data_key, "")
        if "local-files" in url:
            rel = unquote(url.split(f"d={prefix}/", 1)[-1])
            paths.append(_project.sample_file(_project.sample_path_from_mount(rel)))
        else:
            paths.append(Path(url))

    schema = _project.schema
    outputs = _model.predict(paths)
    return {
        "results": [
            {"result": schema.encode_output(output), "score": schema.score(output)}
            for output in outputs
        ]
    }
