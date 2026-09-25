"""A small progress file (`<results>/progress.json`) that `gpubench status` reads.

The run is detached (tmux/nohup) so it survives a dropped SSH or agent session; this file is
how a human or a fresh Claude session sees where it is without parsing logs.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path


class Progress:
    def __init__(self, path: Path):
        self.path = path
        self.state: dict = {}
        if path.exists():
            try:
                self.state = json.loads(path.read_text())
            except ValueError:
                self.state = {}

    def start_campaign(self, campaign: str, sessions: list[str], planned_points: int,
                       estimate_hours: float | None, price_per_hour: float | None) -> None:
        self.state = {
            "campaign": campaign,
            "sessions": sessions,
            "pid": os.getpid(),
            "started_at": _now(),
            "planned_points": planned_points,
            "points_done": 0,
            "estimate_hours": estimate_hours,
            "price_per_hour_usd": price_per_hour,
            "stage": "starting",
            "detail": "",
            "failed_sessions": [],
            "finished": False,
        }
        self._write()

    def update(self, **fields) -> None:
        self.state.update(fields)
        self._write()

    def point_done(self) -> None:
        self.state["points_done"] = self.state.get("points_done", 0) + 1
        self._write()

    def finish(self, failed: list[str]) -> None:
        self.update(stage="finished", detail="", finished=True, failed_sessions=failed,
                    finished_at=_now())

    def _write(self) -> None:
        self.state["updated_at"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.path)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
