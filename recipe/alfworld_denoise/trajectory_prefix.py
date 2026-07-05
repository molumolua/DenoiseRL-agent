import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


ACTION_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class PrefixStep:
    action: str
    observation: Optional[str] = None


@dataclass(frozen=True)
class TrajectoryPrefix:
    task_key: Optional[str]
    steps: tuple[PrefixStep, ...]
    source: Optional[str] = None
    score: Optional[float] = None

    @property
    def actions(self) -> list[str]:
        return [step.action for step in self.steps]

    def to_env_kwargs(self) -> dict[str, Any]:
        return {
            "trajectory_prefix": {
                "actions": self.actions,
                "task_key": self.task_key,
                "source": self.source,
                "score": self.score,
            },
        }


class PrefixPool:
    """Loads small-model ALFWorld trajectories and samples action-level prefixes."""

    def __init__(
        self,
        path: Optional[str],
        *,
        min_steps: int = 1,
        max_steps: Optional[int] = None,
        seed: int = 0,
        require_failed: bool = True,
    ) -> None:
        self.path = path
        self.min_steps = int(min_steps)
        self.max_steps = None if max_steps is None else int(max_steps)
        self.require_failed = bool(require_failed)
        self.rng = random.Random(seed)
        self.by_task: dict[str, list[TrajectoryPrefix]] = {}
        self.all_prefixes: list[TrajectoryPrefix] = []
        if path:
            self._load(Path(path).expanduser())

    def __bool__(self) -> bool:
        return bool(self.all_prefixes)

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Prefix pool file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    prefix = self._record_to_prefix(json.loads(line))
                except Exception as exc:
                    raise ValueError(f"Invalid prefix record at {path}:{line_no}: {exc}") from exc
                if prefix is None:
                    continue
                self.all_prefixes.append(prefix)
                if prefix.task_key:
                    self.by_task.setdefault(prefix.task_key, []).append(prefix)

    def _normalize_action(self, action: str) -> str:
        match = ACTION_RE.search(action)
        if match:
            action = match.group(1)
        return action.strip().lower()

    def _record_to_prefix(self, record: dict[str, Any]) -> Optional[TrajectoryPrefix]:
        if self.require_failed and bool(record.get("success", False)):
            return None
        task_key = record.get("gamefile") or record.get("task_type") or record.get("task_id")

        raw_steps = record.get("steps") or record.get("trajectory") or []
        if not raw_steps and record.get("actions"):
            raw_steps = [{"action": action} for action in record["actions"]]

        steps: list[PrefixStep] = []
        for item in raw_steps:
            if isinstance(item, str):
                action = item
                observation = None
            else:
                action = item.get("action") or item.get("response") or item.get("text")
                observation = item.get("observation") or item.get("obs") or item.get("text_obs")
            if action:
                steps.append(
                    PrefixStep(
                        action=self._normalize_action(str(action)),
                        observation=None if observation is None else str(observation),
                    )
                )

        if len(steps) < self.min_steps:
            return None
        max_len = len(steps) if self.max_steps is None else min(len(steps), self.max_steps)
        cut = self.rng.randint(self.min_steps, max_len)
        return TrajectoryPrefix(
            task_key=None if task_key is None else str(task_key),
            steps=tuple(steps[:cut]),
            source=record.get("source") or record.get("model"),
            score=record.get("score"),
        )

    def sample(self, k: int) -> list[TrajectoryPrefix]:
        if k <= 0 or not self.all_prefixes:
            return []
        return [self.rng.choice(self.all_prefixes) for _ in range(k)]
