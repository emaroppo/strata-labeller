from collections.abc import Callable

from label_studio_sdk import PredictionRequest
from label_studio_sdk.actions.types import CreateActionsRequestSelectedItemsIncluded
from label_studio_sdk.client import LabelStudio

from ..config import Settings
from ..project import LabellingProject
from . import label_config


class LSClient:
    def __init__(self, settings: Settings, project: LabellingProject):
        self.client = LabelStudio(
            base_url=settings.label_studio.url,
            api_key=settings.label_studio.api_key,
            # Bulk imports on large projects can exceed the SDK's 60s default
            timeout=300,
        )
        self.settings = settings
        self.project = project

    def create_project(self, name: str) -> int:
        project = self.client.projects.create(
            title=name,
            label_config=self.project.schema.label_config(),
        )
        if project.id is None:
            raise RuntimeError("Label Studio created a project and answered with no id")
        return project.id

    # ------------------------------------------------------------------
    # Labeling config
    # ------------------------------------------------------------------

    def get_label_config(self, project_id: int) -> str:
        return self.client.projects.get(id=project_id).label_config or ""

    def update_label_config(self, project_id: int, xml: str) -> None:
        self.client.projects.update(id=project_id, label_config=xml)

    def add_class_to_config(self, project_id: int, name: str) -> str:
        """Insert a class into the project's live config, layout untouched."""
        current = self.get_label_config(project_id)
        if not current:
            raise label_config.LabelConfigError(
                f"Label Studio project {project_id} has no labeling config"
            )
        updated = label_config.add_class(current, name)
        self.update_label_config(project_id, updated)
        return updated

    def delete_annotations(self, project_id: int, task_ids: list[int]) -> None:
        """Bulk-delete the annotations on the given tasks.

        Used to bring skipped tasks back into the review queue: the skip is
        recorded as a cancelled annotation, so the task stays out of the
        queue until that annotation is gone.
        """
        if not task_ids:
            return
        self.client.actions.create(
            id="delete_tasks_annotations",
            project=project_id,
            selected_items=CreateActionsRequestSelectedItemsIncluded(all_=False, included=task_ids),
        )

    def clear_predictions(self, project_id: int, task_ids: list[int]) -> None:
        """Take the pre-annotations off the given tasks.

        For a blind second look. See ``docs/adr/0028``.
        """
        if not task_ids:
            return
        self.client.actions.create(
            id="delete_tasks_predictions",
            project=project_id,
            selected_items=CreateActionsRequestSelectedItemsIncluded(all_=False, included=task_ids),
        )

    # ------------------------------------------------------------------
    # Tasks, predictions, annotations
    # ------------------------------------------------------------------

    def setup_local_storage(self, project_id: int, path: str | None = None) -> None:
        self.client.import_storage.local.create(
            project=project_id,
            path=path or self.settings.label_studio.local_storage_path,
            # What is on the mount is whatever the project labels —
            # pictures, documents, frames
            title="Local files",
            use_blob_urls=False,
        )

    # ------------------------------------------------------------------
    # The catalog path
    # ------------------------------------------------------------------

    def import_catalog_tasks(
        self,
        project_id: int,
        tasks: list,
        chunk_size: int = 500,
        on_progress: Callable[[int], None] | None = None,
    ) -> dict[int, int]:
        """Create tasks from :class:`adapter.Task` and map sample id -> task id.

        Chunked, with the map returned as it goes. See ``docs/adr/0013``.
        """
        mapping: dict[int, int] = {}
        for i in range(0, len(tasks), chunk_size):
            batch = tasks[i : i + chunk_size]
            response = self.client.projects.import_tasks(
                id=project_id,
                request=[task.as_import() for task in batch],
                return_task_ids=True,
            )
            task_ids = (response.model_extra or {}).get("task_ids") or []
            if len(task_ids) != len(batch):
                # Without a positional match there is no saying which task is
                # which; --rebuild-map recovers the mapping. docs/adr/0032
                continue
            mapping.update(
                {task.sample_id: task_id for task, task_id in zip(batch, task_ids, strict=True)}
            )
            if on_progress is not None:
                on_progress(len(batch))
        return mapping

    def list_tasks(self, project_id: int) -> list[dict]:
        """Every task, flat enough for the task map to be rebuilt from it."""
        return [
            {"id": task.id, "data": task.data or {}}
            for task in self.client.tasks.list(
                project=project_id, page_size=1000, fields="task_only", resolve_uri=False
            )
        ]

    def update_task_data(self, task_id: int, data: dict) -> None:
        """Repoint one task at a different sample.

        Data is replaced whole rather than merged, which is what the API
        offers — so callers pass the task's existing data with one key
        changed, not just the key.
        """
        self.client.tasks.update(str(task_id), data=data)

    def export_raw(self, project_id: int) -> list[dict]:
        """The export snapshot, unconverted.

        One bulk download rather than paging; what it means is the sync
        layer's business, not this one's.
        """
        return [
            task if isinstance(task, dict) else task.dict()
            for task in self.client.projects.exports.as_json(project_id, timeout=600)
        ]

    def push_catalog_predictions(
        self,
        project_id: int,
        predictions: list[tuple[int, list[dict], float]],
        task_map: dict[int, int],
        model_version: str | None = None,
        replace_existing: bool = False,
        chunk_size: int = 200,
    ) -> int:
        """Bulk-import predictions keyed by sample id."""
        to_push = [
            (task_map[sample_id], results, score)
            for sample_id, results, score in predictions
            if sample_id in task_map
        ]
        if not to_push:
            return 0

        if replace_existing:
            self.client.actions.create(
                id="delete_tasks_predictions",
                project=project_id,
                selected_items=CreateActionsRequestSelectedItemsIncluded(
                    all_=False, included=[t for t, _, _ in to_push]
                ),
            )

        requests = [
            PredictionRequest(
                task=task_id, result=results, score=score, model_version=model_version
            )
            for task_id, results, score in to_push
        ]
        for i in range(0, len(requests), chunk_size):
            self.client.projects.import_predictions(
                id=project_id, request=requests[i : i + chunk_size]
            )

        if model_version is not None:
            # LS only displays predictions matching the project's active
            # model version, so keep it pointed at what was just pushed
            self.client.projects.update(id=project_id, model_version=model_version)
        return len(to_push)
