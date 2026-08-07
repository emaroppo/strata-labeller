from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from .schemas import Prediction

__all__ = ["BaseModel", "Prediction"]


class BaseModel(ABC):
    """Subclass this and implement the four methods below.

    Your model (e.g. a ``torch.nn.Module``) should be stored as an
    instance attribute so that ``finetune`` and ``predict`` can access it.

    Targets and outputs are the schema's own types rather than Label Studio
    dictionaries: a classifier sees ``list[str]`` and returns a
    ``ChoiceOutput``, a detector sees ``list[Box]`` and returns a
    ``BoxOutput``. The schema converts to and from Label Studio.
    """

    #: Which schema this model is written for; checked before training
    schema_type: ClassVar[str] = "image_classification"

    @abstractmethod
    def finetune(
        self,
        samples: list[dict],
        classes: list[str],
        val_samples: list[dict] | None = None,
    ) -> dict:
        """Fine-tune on labeled data.

        ``samples`` is a list of dicts, each with keys ``"path"`` (an
        absolute path to the sample's file, whether image or document) and
        ``"target"`` (whatever the schema decodes to).
        ``classes`` is the full class list. ``val_samples`` (same shape) is
        held-out data: evaluate on it after training and include the
        results in the metrics dict.

        Return a metrics dict, e.g.
        ``{"loss": 0.12, "accuracy": 0.93, "val_loss": 0.2, "val_accuracy": 0.9}``.
        """
        ...

    @abstractmethod
    def predict(self, paths: list[Path]) -> list[Any]:
        """Return one schema output per file path, in order."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        ...
