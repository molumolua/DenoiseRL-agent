"""Fresh-gamefile, per-task-type noise curriculum for ALFWorld DenoiseRL v2."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _normalize_id(value):
    return value.item() if isinstance(value, np.generic) else value


def _metric_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "unknown"


@dataclass
class TaskTypeNoiseState:
    rho: float
    num_updates: int = 0
    gamefiles_seen: int = 0
    last_accuracy: float | None = None


class TaskTypePoolCurriculum:
    """Shuffle gamefiles by epoch and maintain one noise ratio per task type.

    Every active gamefile is used for exactly one optimizer step. After the step,
    all active rows are replaced by the next unseen rows in the shuffled epoch.
    The final partial epoch is joined with the start of the next shuffled epoch so
    the rollout batch size stays constant.
    """

    STATE_VERSION = 3

    def __init__(
        self,
        problem_ids: Sequence,
        problem_id_to_task_type: Mapping,
        batch_size: int,
        *,
        initial_rho: float = 0.0,
        min_rho: float = 0.0,
        max_rho: float = 0.5,
        target_accuracy: float = 0.75,
        alpha: float = 0.2,
        shuffle_seed: int = 0,
    ) -> None:
        self.problem_ids = tuple(_normalize_id(pid) for pid in problem_ids)
        if not self.problem_ids:
            raise ValueError("DenoiseRL v2 requires a non-empty training pool.")
        try:
            unique_problem_ids = set(self.problem_ids)
        except TypeError as exc:
            raise ValueError("Every problem_id must be hashable.") from exc
        if len(unique_problem_ids) != len(self.problem_ids):
            raise ValueError("DenoiseRL v2 requires a unique problem_id per pool row.")
        self.problem_id_to_index = {
            problem_id: index for index, problem_id in enumerate(self.problem_ids)
        }

        normalized_task_types = {
            _normalize_id(problem_id): str(task_type)
            for problem_id, task_type in problem_id_to_task_type.items()
        }
        missing_task_types = unique_problem_ids - set(normalized_task_types)
        if missing_task_types:
            raise ValueError(
                "Missing task type for DenoiseRL v2 gamefile(s): "
                f"{len(missing_task_types)} missing."
            )
        self.problem_id_to_task_type = {
            problem_id: normalized_task_types[problem_id]
            for problem_id in self.problem_ids
        }
        if any(not task_type for task_type in self.problem_id_to_task_type.values()):
            raise ValueError("Every DenoiseRL v2 task type must be non-empty.")
        self.task_types = tuple(sorted(set(self.problem_id_to_task_type.values())))

        self.batch_size = int(batch_size)
        if not (1 <= self.batch_size <= len(self.problem_ids)):
            raise ValueError(
                "batch_size must be in [1, pool_size], got "
                f"{self.batch_size} for pool_size={len(self.problem_ids)}."
            )

        self.min_rho = self._finite("min_rho", min_rho)
        self.max_rho = self._finite("max_rho", max_rho)
        self.initial_rho = self._finite("initial_rho", initial_rho)
        if not (0.0 <= self.min_rho <= self.max_rho <= 1.0):
            raise ValueError("rho bounds must satisfy 0 <= min_rho <= max_rho <= 1.")
        if not (self.min_rho <= self.initial_rho <= self.max_rho):
            raise ValueError("initial_rho must lie within the configured rho bounds.")

        self.target_accuracy = self._finite("target_accuracy", target_accuracy)
        if not (0.0 <= self.target_accuracy <= 1.0):
            raise ValueError("target_accuracy must be in [0, 1].")
        self.alpha = self._finite("alpha", alpha)
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative.")

        self.shuffle_seed = int(shuffle_seed)
        self.shuffle_rng = np.random.default_rng(self.shuffle_seed)
        self.task_states = {
            task_type: TaskTypeNoiseState(rho=self.initial_rho)
            for task_type in self.task_types
        }

        self.pool_order: list[int] = []
        self.pool_cursor = 0
        self.pool_epoch = 0
        self.epochs_completed = 0
        self.steps_completed = 0
        self.gamefiles_trained_total = 0
        self.active_indices: list[int] = []
        self._activate_next_batch(previous_active=())

    @staticmethod
    def _finite(name: str, value) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite, got {value!r}.")
        return result

    @property
    def pool_size(self) -> int:
        return len(self.problem_ids)

    @property
    def active_problem_ids(self) -> tuple:
        return tuple(self.problem_ids[index] for index in self.active_indices)

    @property
    def active_task_types(self) -> tuple[str, ...]:
        return tuple(
            self.task_type_for_problem(problem_id)
            for problem_id in self.active_problem_ids
        )

    def task_type_for_problem(self, problem_id) -> str:
        normalized = _normalize_id(problem_id)
        if normalized not in self.problem_id_to_task_type:
            raise KeyError(f"Unknown DenoiseRL v2 problem_id {normalized!r}.")
        return self.problem_id_to_task_type[normalized]

    def rho_for_task_type(self, task_type: str) -> float:
        normalized = str(task_type)
        if normalized not in self.task_states:
            raise KeyError(f"Unknown DenoiseRL v2 task type {normalized!r}.")
        return self.task_states[normalized].rho

    def rho_for_problem(self, problem_id) -> float:
        return self.rho_for_task_type(self.task_type_for_problem(problem_id))

    def mean_rho(self) -> float:
        return float(np.mean([state.rho for state in self.task_states.values()]))

    def _start_new_epoch(
        self,
        exclude_prefix: set[int],
        avoid_prefix: set[int],
        prefix_len: int,
    ) -> None:
        order = [int(index) for index in self.shuffle_rng.permutation(self.pool_size)]
        if prefix_len > 0:
            preferred = [
                index
                for index in order
                if index not in exclude_prefix and index not in avoid_prefix
            ]
            reusable_previous = [
                index
                for index in order
                if index not in exclude_prefix and index in avoid_prefix
            ]
            excluded = [index for index in order if index in exclude_prefix]
            eligible = preferred + reusable_previous
            if len(eligible) < prefix_len:
                raise RuntimeError(
                    "DenoiseRL v2 cannot fill a unique active batch from its pool."
                )
            order = eligible + excluded

        if self.pool_epoch > 0:
            self.epochs_completed += 1
        self.pool_epoch += 1
        self.pool_order = order
        self.pool_cursor = 0

    def _activate_next_batch(self, previous_active: Sequence[int]) -> None:
        selected: list[int] = []
        previous = set(int(index) for index in previous_active)
        while len(selected) < self.batch_size:
            if self.pool_cursor >= len(self.pool_order):
                needed = self.batch_size - len(selected)
                self._start_new_epoch(
                    exclude_prefix=set(selected),
                    avoid_prefix=previous,
                    prefix_len=needed,
                )
            take = min(
                self.batch_size - len(selected),
                self.pool_size - self.pool_cursor,
            )
            selected.extend(self.pool_order[self.pool_cursor : self.pool_cursor + take])
            self.pool_cursor += take

        if len(set(selected)) != len(selected):
            raise RuntimeError("DenoiseRL v2 produced a duplicate gamefile in one batch.")
        self.active_indices = selected

    def _pool_metrics(self) -> dict[str, float]:
        active_rhos = [
            self.rho_for_problem(problem_id)
            for problem_id in self.active_problem_ids
        ]
        return {
            "denoise/v2/enabled": 1.0,
            "denoise/v2/active_batch_size": float(len(self.active_indices)),
            "denoise/v2/pool_size": float(self.pool_size),
            "denoise/v2/pool_cursor": float(self.pool_cursor),
            "denoise/v2/pool_remaining": float(self.pool_size - self.pool_cursor),
            "denoise/v2/coverage_fraction_this_epoch": float(
                self.pool_cursor / self.pool_size
            ),
            "denoise/v2/pool_epoch": float(self.pool_epoch),
            "denoise/v2/epochs_completed": float(self.epochs_completed),
            "denoise/v2/steps_completed": float(self.steps_completed),
            "denoise/v2/gamefiles_trained_total": float(self.gamefiles_trained_total),
            "denoise/v2/rho_active_mean": float(np.mean(active_rhos)),
            "denoise/v2/rho_active_min": float(np.min(active_rhos)),
            "denoise/v2/rho_active_max": float(np.max(active_rhos)),
        }

    def update(self, problem_id_to_average_accuracy: Mapping) -> dict[str, float]:
        """Update task-type rhos from this fresh batch, then replace every row."""
        accuracies = {
            _normalize_id(problem_id): accuracy
            for problem_id, accuracy in problem_id_to_average_accuracy.items()
        }
        active_problem_ids = self.active_problem_ids
        expected = set(active_problem_ids)
        observed = set(accuracies)
        if observed != expected:
            raise ValueError(
                "DenoiseRL v2 accuracies must match the active gamefile batch: "
                f"missing={len(expected - observed)}, unexpected={len(observed - expected)}."
            )

        by_task_type: dict[str, list[float]] = {
            task_type: [] for task_type in self.task_types
        }
        used_rhos: list[float] = []
        all_accuracies: list[float] = []
        for problem_id in active_problem_ids:
            accuracy = self._finite(
                f"average accuracy for problem_id {problem_id!r}",
                accuracies[problem_id],
            )
            if not (0.0 <= accuracy <= 1.0):
                raise ValueError(
                    f"Average accuracy for problem_id {problem_id!r} must be in [0, 1]."
                )
            task_type = self.task_type_for_problem(problem_id)
            by_task_type[task_type].append(accuracy)
            used_rhos.append(self.rho_for_task_type(task_type))
            all_accuracies.append(accuracy)

        task_metrics: dict[str, float] = {}
        updated_task_types = 0
        for task_type, task_accuracies in by_task_type.items():
            state = self.task_states[task_type]
            slug = _metric_slug(task_type)
            if task_accuracies:
                average_accuracy = float(np.mean(task_accuracies))
                state.last_accuracy = average_accuracy
                state.rho = min(
                    self.max_rho,
                    max(
                        self.min_rho,
                        state.rho
                        + self.alpha * (average_accuracy - self.target_accuracy),
                    ),
                )
                state.num_updates += 1
                state.gamefiles_seen += len(task_accuracies)
                updated_task_types += 1
                task_metrics[f"denoise/v2/task_type/{slug}/accuracy"] = average_accuracy
                task_metrics[f"denoise/v2/task_type/{slug}/batch_gamefiles"] = float(
                    len(task_accuracies)
                )
            task_metrics[f"denoise/v2/task_type/{slug}/rho"] = state.rho
            task_metrics[f"denoise/v2/task_type/{slug}/updates"] = float(state.num_updates)
            task_metrics[f"denoise/v2/task_type/{slug}/gamefiles_seen"] = float(
                state.gamefiles_seen
            )

        previous_active = tuple(self.active_indices)
        previous_epoch = self.pool_epoch
        self.steps_completed += 1
        self.gamefiles_trained_total += len(previous_active)
        self._activate_next_batch(previous_active=previous_active)

        metrics = self._pool_metrics()
        metrics.update(
            {
                "denoise/v2/accuracy_mean": float(np.mean(all_accuracies)),
                "denoise/v2/rho_used_mean": float(np.mean(used_rhos)),
                "denoise/v2/task_types_updated": float(updated_task_types),
                "denoise/v2/replaced_this_step": float(len(previous_active)),
                "denoise/v2/epoch_advanced_this_step": float(
                    self.pool_epoch != previous_epoch
                ),
            }
        )
        metrics.update(task_metrics)
        return metrics

    def metrics(self) -> dict[str, float]:
        metrics = self._pool_metrics()
        for task_type, state in self.task_states.items():
            slug = _metric_slug(task_type)
            metrics[f"denoise/v2/task_type/{slug}/rho"] = state.rho
            metrics[f"denoise/v2/task_type/{slug}/updates"] = float(state.num_updates)
            metrics[f"denoise/v2/task_type/{slug}/gamefiles_seen"] = float(
                state.gamefiles_seen
            )
            if state.last_accuracy is not None:
                metrics[f"denoise/v2/task_type/{slug}/last_accuracy"] = state.last_accuracy
        return metrics

    def state_dict(self) -> dict:
        return {
            "version": self.STATE_VERSION,
            "pool_size": self.pool_size,
            "problem_id_reprs": [repr(problem_id) for problem_id in self.problem_ids],
            "problem_task_types": [
                self.problem_id_to_task_type[problem_id] for problem_id in self.problem_ids
            ],
            "batch_size": self.batch_size,
            "shuffle_seed": self.shuffle_seed,
            "pool_order": list(self.pool_order),
            "pool_cursor": self.pool_cursor,
            "active_indices": list(self.active_indices),
            "pool_epoch": self.pool_epoch,
            "epochs_completed": self.epochs_completed,
            "steps_completed": self.steps_completed,
            "gamefiles_trained_total": self.gamefiles_trained_total,
            "task_states": {
                task_type: {
                    "rho": state.rho,
                    "num_updates": state.num_updates,
                    "gamefiles_seen": state.gamefiles_seen,
                    "last_accuracy": state.last_accuracy,
                }
                for task_type, state in self.task_states.items()
            },
            "shuffle_rng_state": self.shuffle_rng.bit_generator.state,
        }

    def load_state_dict(self, saved: Mapping) -> None:
        version = int(saved.get("version", -1))
        if version != self.STATE_VERSION:
            raise ValueError(
                "Unsupported DenoiseRL v2 curriculum state version "
                f"{version}; task-type curriculum requires version {self.STATE_VERSION}."
            )
        if int(saved.get("pool_size", -1)) != self.pool_size:
            raise ValueError("Curriculum checkpoint pool size does not match the dataset.")
        if saved.get("problem_id_reprs") != [repr(pid) for pid in self.problem_ids]:
            raise ValueError("Curriculum checkpoint pool order does not match the dataset.")
        expected_task_types = [
            self.problem_id_to_task_type[problem_id] for problem_id in self.problem_ids
        ]
        if saved.get("problem_task_types") != expected_task_types:
            raise ValueError("Curriculum checkpoint task types do not match the dataset.")
        if int(saved.get("batch_size", -1)) != self.batch_size:
            raise ValueError("Curriculum checkpoint batch size does not match the config.")
        if int(saved.get("shuffle_seed", self.shuffle_seed)) != self.shuffle_seed:
            raise ValueError("Curriculum checkpoint shuffle seed does not match the config.")

        pool_order = [int(index) for index in saved["pool_order"]]
        if sorted(pool_order) != list(range(self.pool_size)):
            raise ValueError("Invalid shuffled pool order in curriculum checkpoint.")
        pool_cursor = int(saved["pool_cursor"])
        if not (0 <= pool_cursor <= self.pool_size):
            raise ValueError("Invalid pool cursor in curriculum checkpoint.")
        active_indices = [int(index) for index in saved["active_indices"]]
        if len(active_indices) != self.batch_size or len(set(active_indices)) != self.batch_size:
            raise ValueError("Invalid active batch in curriculum checkpoint.")
        if any(index < 0 or index >= self.pool_size for index in active_indices):
            raise ValueError("Out-of-range active index in curriculum checkpoint.")

        task_states: dict[str, TaskTypeNoiseState] = {}
        raw_task_states = saved.get("task_states", {})
        if set(raw_task_states) != set(self.task_types):
            raise ValueError("Curriculum checkpoint task-type states do not match the dataset.")
        for task_type in self.task_types:
            raw_state = raw_task_states[task_type]
            rho = self._finite("checkpoint rho", raw_state["rho"])
            if not (self.min_rho <= rho <= self.max_rho):
                raise ValueError("Checkpoint rho is outside the configured bounds.")
            last_accuracy = raw_state.get("last_accuracy")
            if last_accuracy is not None:
                last_accuracy = self._finite("checkpoint last_accuracy", last_accuracy)
                if not (0.0 <= last_accuracy <= 1.0):
                    raise ValueError("Checkpoint last_accuracy must be in [0, 1].")
            task_states[task_type] = TaskTypeNoiseState(
                rho=rho,
                num_updates=int(raw_state["num_updates"]),
                gamefiles_seen=int(raw_state["gamefiles_seen"]),
                last_accuracy=last_accuracy,
            )

        self.pool_order = pool_order
        self.pool_cursor = pool_cursor
        self.active_indices = active_indices
        self.pool_epoch = int(saved["pool_epoch"])
        self.epochs_completed = int(saved["epochs_completed"])
        self.steps_completed = int(saved["steps_completed"])
        self.gamefiles_trained_total = int(saved["gamefiles_trained_total"])
        self.task_states = task_states
        self.shuffle_rng.bit_generator.state = saved["shuffle_rng_state"]


# Keep the old import name available to downstream code while changing its
# semantics from per-gamefile state to per-task-type state.
GamefilePoolCurriculum = TaskTypePoolCurriculum
