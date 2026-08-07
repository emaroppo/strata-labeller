import json
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote, unquote

from label_studio_sdk import PredictionRequest
from label_studio_sdk.client import LabelStudio

from . import label_config
from .config import Settings
from .dataset import Sample
from .project import Project
from .schemas import Prediction


class LSClient:
    def __init__(self, settings: Settings, project: Project):
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
            selected_items={"all": False, "included": task_ids},
        )

    # ------------------------------------------------------------------
    # Image URL mapping
    # ------------------------------------------------------------------

    def _sample_url(self, sample_path: str) -> str:
        # Percent-encode so characters like '&' or '#' don't break the query string
        rel = self.project.mount_relative_path(sample_path)
        prefix = self.project.label_studio.local_files_prefix
        return f"/data/local-files/?d={prefix}/{quote(rel)}"

    def _url_to_path(self, image_url: str) -> str:
        if "local-files" not in image_url:
            return image_url
        prefix = self.project.label_studio.local_files_prefix
        rel = unquote(image_url.split(f"d={prefix}/", 1)[-1])
        return self.project.sample_path_from_mount(rel)

    # ------------------------------------------------------------------
    # Tasks, predictions, annotations
    # ------------------------------------------------------------------

    def import_tasks(self, project_id: int, samples: list[Sample]) -> None:
        data_key = self.project.schema.data_key
        tasks: list[dict] = []
        for s in samples:
            task: dict = {"data": {data_key: self._sample_url(s.path)}}
            if s.is_labeled:
                # Already stored in Label Studio's own shape
                task["annotations"] = [{"result": s.results}]
            tasks.append(task)
        resp = self.client.projects.import_tasks(
            id=project_id, request=tasks, return_task_ids=True
        )
        # task_ids is untyped but present with return_task_ids; keep the
        # local cache current so the next push needs no full task listing
        task_ids = (resp.model_extra or {}).get("task_ids") or []
        cache = self._load_task_map_cache(project_id)
        if cache is not None and len(task_ids) == len(samples):
            for s, task_id in zip(samples, task_ids):
                cache[s.path] = [task_id, False]
            self._save_task_map_cache(project_id, cache)

    def push_predictions(
        self,
        project_id: int,
        predictions: list[Prediction],
        task_id_map: dict[str, int],
        model_version: str | None = None,
        replace_existing: bool = False,
        on_progress: Callable[[int], None] | None = None,
        chunk_size: int = 200,
    ) -> int:
        """Bulk-import predictions. Returns the number pushed.

        With replace_existing, old predictions on the target tasks are
        bulk-deleted first instead of accumulating duplicates.
        """
        to_push = [
            (pred, task_id_map[pred.path])
            for pred in predictions
            if pred.path in task_id_map
        ]
        if not to_push:
            return 0

        if replace_existing:
            self.client.actions.create(
                id="delete_tasks_predictions",
                project=project_id,
                selected_items={"all": False, "included": [tid for _, tid in to_push]},
            )

        requests = [
            PredictionRequest(
                task=task_id,
                result=pred.results,
                score=pred.score,
                model_version=model_version,
            )
            for pred, task_id in to_push
        ]
        for i in range(0, len(requests), chunk_size):
            chunk = requests[i : i + chunk_size]
            self.client.projects.import_predictions(id=project_id, request=chunk)
            if on_progress is not None:
                on_progress(len(chunk))

        if model_version is not None:
            # LS only displays predictions whose version matches the project's
            # active model version, so keep it pointed at what we just pushed
            self.client.projects.update(id=project_id, model_version=model_version)

        cache = self._load_task_map_cache(project_id)
        if cache is not None:
            for pred, _task_id in to_push:
                if pred.path in cache:
                    cache[pred.path][1] = True
            self._save_task_map_cache(project_id, cache)
        return len(to_push)

    def export_annotations(self, project_id: int) -> list[Sample]:
        # Snapshot export is one bulk download instead of paging through tasks;
        # the timeout is how long we wait for LS to prepare the snapshot
        schema = self.project.schema
        tasks = self.client.projects.exports.as_json(project_id, timeout=600)
        samples: list[Sample] = []
        for task in tasks:
            sample_path = self._url_to_path(task.get("data", {}).get(schema.data_key, ""))

            results = []
            annotated = False
            skipped = False
            annotations = task.get("annotations") or []
            if annotations:
                latest = annotations[-1]
                if latest.get("was_cancelled"):
                    # Skipped: reviewed but nothing applies. The cancelled
                    # annotation snapshots the pre-annotation (a model guess),
                    # so its contents are not an answer.
                    skipped = True
                else:
                    results = schema.canonicalize(latest.get("result") or [])
                    annotated = True

            samples.append(
                Sample(
                    path=sample_path,
                    results=results,
                    annotated=annotated,
                    skipped=skipped,
                )
            )
        return samples

    # ------------------------------------------------------------------
    # Task-id map + local cache
    #
    # Listing every task takes minutes on large projects, so we keep a
    # local {path: [task_id, has_prediction]} cache in the project's
    # .state/ directory. It is extended on import, updated on push, and
    # rebuilt whenever the project's task count no longer matches (e.g.
    # tasks added/deleted in the LS UI). Prediction changes made manually
    # in LS are NOT detected; pass use_cache=False (CLI: --no-cache) to
    # force a full refetch.
    # ------------------------------------------------------------------

    def _task_map_cache_path(self, project_id: int) -> Path:
        return self.project.state_dir / f"task_map_{project_id}.json"

    def _load_task_map_cache(self, project_id: int) -> dict[str, list] | None:
        path = self._task_map_cache_path(project_id)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def _save_task_map_cache(self, project_id: int, cache: dict[str, list]) -> None:
        path = self._task_map_cache_path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(cache, f)

    def get_task_id_map(
        self,
        project_id: int,
        exclude_predicted: bool = False,
        use_cache: bool = True,
    ) -> dict[str, int]:
        if use_cache:
            cache = self._load_task_map_cache(project_id)
            if cache is not None:
                total = self.client.projects.get(id=project_id).task_number or 0
                if total == len(cache):
                    return {
                        path: entry[0]
                        for path, entry in cache.items()
                        if not (exclude_predicted and entry[1])
                    }

        tasks = self.client.tasks.list(
            project=project_id,
            page_size=1000,
            fields="task_only",
            resolve_uri=False,
        )
        cache = {}
        for task in tasks:
            path = self._url_to_path(task.data.get("image", ""))
            cache[path] = [task.id, (task.total_predictions or 0) > 0]
        self._save_task_map_cache(project_id, cache)

        return {
            path: entry[0]
            for path, entry in cache.items()
            if not (exclude_predicted and entry[1])
        }

    def setup_local_storage(self, project_id: int, path: str | None = None) -> None:
        self.client.import_storage.local.create(
            project=project_id,
            path=path or self.settings.label_studio.local_storage_path,
            title="Local Images",
            use_blob_urls=False,
        )

    # ------------------------------------------------------------------
    # The catalog path
    # ------------------------------------------------------------------

    def import_catalog_tasks(self, project_id: int, tasks: list) -> dict[int, int]:
        """Create tasks from :class:`adapter.Task` and map sample id -> task id.

        Returns the mapping rather than caching it here: the catalog knows
        what a sample is, and this class should not.
        """
        if not tasks:
            return {}
        response = self.client.projects.import_tasks(
            id=project_id,
            request=[task.as_import() for task in tasks],
            return_task_ids=True,
        )
        task_ids = (response.model_extra or {}).get("task_ids") or []
        if len(task_ids) != len(tasks):
            # Without a positional match there is no way to say which task
            # is which, and guessing would corrupt the map
            return {}
        return {task.sample_id: task_id for task, task_id in zip(tasks, task_ids)}

    def list_tasks(self, project_id: int) -> list[dict]:
        """Every task, flat enough for the task map to be rebuilt from it."""
        return [
            {"id": task.id, "data": task.data or {}}
            for task in self.client.tasks.list(
                project=project_id, page_size=1000, fields="task_only", resolve_uri=False
            )
        ]

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
                selected_items={"all": False, "included": [t for t, _, _ in to_push]},
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

    def add_ml_backend(
        self,
        project_id: int,
        url: str = "http://host.docker.internal:9090",
        name: str = "auto-labeller",
    ) -> None:
        self.client.ml.create(project=project_id, url=url, title=name)
