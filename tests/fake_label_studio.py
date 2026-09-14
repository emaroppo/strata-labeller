"""A Label Studio that lives in a dict.

Stands in for :class:`ls_client.LSClient`, which is the only thing between
the round loop and a server nothing in a test can reach. That gap is why
`push` has broken three times without a failing test: a stale import after
a module moved, a type mismatch between the cache and the local handler,
and a run passed around as a dict that two lines still read as an object.
Each was found by running it, on whichever machine ran it first.

Faithful where it matters and lazy where it does not. It records what it
was asked to create, keeps the tasks so they can be listed back, and hands
out task ids positionally, because a caller that cannot match tasks to
samples corrupts its map — which is the one thing the real client is
careful about.
"""

from dataclasses import dataclass, field


@dataclass
class FakeLabelStudio:
    """Enough Label Studio to run a round against."""

    projects: dict[int, str] = field(default_factory=dict)
    #: task id -> {"project", "data", "annotations"}
    tasks: dict[int, dict] = field(default_factory=dict)
    #: task id -> list of predictions pushed for it
    predictions: dict[int, list] = field(default_factory=dict)
    storage: list[tuple[int, str]] = field(default_factory=list)
    deleted_annotations: list[int] = field(default_factory=list)
    _next_project: int = 1
    _next_task: int = 1

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def create_project(self, name: str) -> int:
        project_id = self._next_project
        self._next_project += 1
        self.projects[project_id] = name
        return project_id

    def get_label_config(self, project_id: int) -> str:
        return "<View></View>"

    def update_label_config(self, project_id: int, xml: str) -> None:
        pass

    def add_class_to_config(self, project_id: int, name: str) -> str:
        return "<View></View>"

    def setup_local_storage(self, project_id: int, path: str | None = None) -> None:
        self.storage.append((project_id, path or ""))

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def import_catalog_tasks(
        self, project_id: int, tasks: list, chunk_size: int = 500, on_progress=None
    ) -> dict[int, int]:
        mapping = {}
        for task in tasks:
            task_id = self._next_task
            self._next_task += 1
            payload = task.as_import()
            self.tasks[task_id] = {
                "project": project_id,
                "data": payload["data"],
                "annotations": payload.get("annotations", []),
            }
            mapping[task.sample_id] = task_id
        if on_progress is not None and tasks:
            on_progress(len(tasks))
        return mapping

    def list_tasks(self, project_id: int) -> list[dict]:
        return [
            {"id": task_id, "data": task["data"]}
            for task_id, task in self.tasks.items()
            if task["project"] == project_id
        ]

    def update_task_data(self, task_id: int, data: dict) -> None:
        self.tasks[task_id]["data"] = data

    def export_raw(self, project_id: int) -> list[dict]:
        return [
            {
                "id": task_id,
                "data": task["data"],
                "annotations": task["annotations"],
            }
            for task_id, task in self.tasks.items()
            if task["project"] == project_id
        ]

    def delete_annotations(self, project_id: int, task_ids: list[int]) -> None:
        self.deleted_annotations.extend(task_ids)
        for task_id in task_ids:
            if task_id in self.tasks:
                self.tasks[task_id]["annotations"] = []

    def clear_predictions(self, project_id: int, task_ids: list[int]) -> None:
        for task_id in task_ids:
            self.predictions.pop(task_id, None)

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def push_catalog_predictions(
        self,
        project_id: int,
        predictions: list[tuple[int, list[dict], float]],
        task_map: dict[int, int],
        model_version: str | None = None,
        replace_existing: bool = False,
        chunk_size: int = 200,
    ) -> int:
        # Mapped through task_map exactly as the real one does: a sample with
        # no task is silently skipped there, and a test should see that.
        pushed = 0
        for sample_id, results, score in predictions:
            task_id = task_map.get(sample_id)
            if task_id is None:
                continue
            if replace_existing:
                self.predictions[task_id] = []
            self.predictions.setdefault(task_id, []).append(
                {"result": results, "score": score, "model_version": model_version}
            )
            pushed += 1
        return pushed

    # ------------------------------------------------------------------
    # For assertions
    # ------------------------------------------------------------------

    def answer(self, task_id: int, results: list[dict]) -> None:
        """Stand in for a human submitting an annotation."""
        self.tasks[task_id]["annotations"] = [{"was_cancelled": False, "result": results}]

    def skip(self, task_id: int) -> None:
        self.tasks[task_id]["annotations"] = [{"was_cancelled": True, "result": []}]
