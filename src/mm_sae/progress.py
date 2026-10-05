"""Persistent progress and measured task ETAs, independent of any research question."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import shutil
import resource
import sys
from contextlib import contextmanager
from collections.abc import Sized
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

from .io import atomic_json

_active: ContextVar[ProgressReporter | None] = ContextVar("progress_reporter", default=None)
T = TypeVar("T")


@dataclass
class TaskProgress:
    label: str
    total: int | None
    unit: str
    completed: int = 0
    initial: int = 0
    started: float = field(default_factory=time.monotonic)
    details: dict[str, Any] = field(default_factory=dict)

    def update(self, completed: int, **details):
        self.completed = int(completed)
        self.details.update(details)

    def snapshot(self):
        elapsed = max(0.0, time.monotonic() - self.started)
        processed = self.completed - self.initial
        speed = processed / elapsed if elapsed > 0 and processed > 0 else None
        remaining = max(0, self.total - self.completed) if self.total is not None else None
        return {
            "label": self.label,
            "unit": self.unit,
            "completed": self.completed,
            "total": self.total,
            "percent": 100 * self.completed / self.total if self.total else None,
            "elapsed_seconds": elapsed,
            "units_per_second": speed,
            "eta_seconds": 0
            if remaining == 0
            else remaining / speed
            if speed is not None and remaining is not None
            else None,
            "eta_basis": "observed average rate in this task; approximate for unequal-cost items",
            "details": self.details.copy(),
        }


class ProgressReporter:
    def __init__(self, root: Path, stages: list[str], completed: list[str], interval: float = 10):
        self.root, self.stages, self.completed = root, stages, completed.copy()
        self.interval = interval
        self.tasks: list[TaskProgress] = []
        self.stage: str | None = None
        self.stage_started: float | None = None
        self.started = time.time()
        self.state = "running"
        self.error: str | None = None
        self._stop = threading.Event()
        self._mutex = threading.RLock()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)

    def write(self):
        with self._mutex:
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            gpu = None
            # A heartbeat can observe torch while another thread is still importing it.
            cuda = getattr(sys.modules.get("torch"), "cuda", None)
            is_initialized = getattr(cuda, "is_initialized", None)
            if cuda is not None and callable(is_initialized) and is_initialized():
                gpu = {
                    "allocated_bytes": cuda.memory_allocated(),
                    "reserved_bytes": cuda.memory_reserved(),
                }
            atomic_json(
                self.root / "progress.json",
                {
                    "state": self.state,
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "started_at": self.started,
                    "updated_at": time.time(),
                    "elapsed_seconds": time.time() - self.started,
                    "current_stage": self.stage,
                    "stages_completed": self.completed.copy(),
                    "stages_planned": self.stages,
                    "stage_elapsed_seconds": time.monotonic() - self.stage_started
                    if self.stage_started is not None
                    else None,
                    "tasks": [task.snapshot() for task in self.tasks],
                    "heartbeat_interval_seconds": self.interval,
                    "whole_run_eta_seconds": 0 if self.state == "completed" else None,
                    "whole_run_eta_note": "All planned stages complete."
                    if self.state == "completed"
                    else "Unmeasured stages have unknown runtime. Task ETAs are shown separately.",
                    "error": self.error,
                    "disk_free_bytes": shutil.disk_usage(self.root).free,
                    "peak_process_rss_bytes": peak_rss if sys.platform == "darwin" else peak_rss * 1024,
                    "gpu_allocator": gpu,
                },
            )

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            self.write()

    def __enter__(self):
        self._token = _active.set(self)
        self.write()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join(timeout=self.interval + 1)
        self.state = (
            "failed"
            if exc is not None
            else "completed"
            if set(self.stages) <= set(self.completed)
            else "paused"
        )
        self.error = str(exc) if exc is not None else None
        self.write()
        _active.reset(self._token)


@contextmanager
def stage_progress(name: str):
    reporter = _active.get()
    if reporter is not None:
        reporter.stage, reporter.stage_started = name, time.monotonic()
        reporter.write()
    try:
        yield
    except BaseException:
        raise
    else:
        if reporter is not None:
            reporter.completed.append(name)
            reporter.write()


@contextmanager
def progress_task(label: str, total: int | None = None, unit: str = "items", initial: int = 0):
    reporter = _active.get()
    task = TaskProgress(label, total, unit, completed=initial, initial=initial)
    if reporter is not None:
        with reporter._mutex:
            reporter.tasks.append(task)
        reporter.write()
    try:
        yield task
    finally:
        if reporter is not None:
            with reporter._mutex:
                reporter.tasks.remove(task)
            reporter.write()


def iter_progress(
    items: Iterable[T], label: str, total: int | None = None, unit: str = "items", initial: int = 0
) -> Iterator[T]:
    if total is None and isinstance(items, Sized):
        total = len(items)
    with progress_task(label, total, unit, initial=initial) as task:
        for index, item in enumerate(items, start=initial):
            yield item
            task.update(index + 1)


def read_status(root: Path) -> dict[str, Any]:
    path = root / "progress.json"
    if not path.exists():
        return {"state": "not_started", "output": str(root)}
    value = json.loads(path.read_text())
    value["heartbeat_age_seconds"] = max(0, time.time() - value["updated_at"])
    if value["state"] == "running" and value["hostname"] == socket.gethostname():
        try:
            os.kill(value["pid"], 0)
        except ProcessLookupError:
            value["state"] = "process_gone"
        except PermissionError:
            pass
    if value["state"] == "running" and value["heartbeat_age_seconds"] > max(
        60, value["heartbeat_interval_seconds"] * 3
    ):
        value["state"] = "heartbeat_stale"
    return value


def format_status(value: dict) -> str:
    def duration(seconds):
        if seconds is None:
            return "unknown"
        return f"{int(seconds) // 3600}h {int(seconds) % 3600 // 60}m {int(seconds) % 60}s"

    lines = [f"State: {value['state']}"]
    if "current_stage" not in value:
        return "\n".join(lines)
    lines += [
        f"Stage: {value['current_stage']} ({len(value['stages_completed'])}/{len(value['stages_planned'])} stages complete)",
        f"Elapsed: {duration(value['elapsed_seconds'])}; heartbeat age: {value['heartbeat_age_seconds']:.1f}s",
    ]
    lines.append(
        f"Disk free: {value['disk_free_bytes'] / 1024**3:.1f} GiB; peak process RAM: {value['peak_process_rss_bytes'] / 1024**3:.2f} GiB"
    )
    if value["gpu_allocator"] is not None:
        lines.append(
            f"GPU tensor memory: {value['gpu_allocator']['allocated_bytes'] / 1024**3:.2f} GiB; reserved: {value['gpu_allocator']['reserved_bytes'] / 1024**3:.2f} GiB"
        )
    for task in value["tasks"]:
        lines.append(
            f"{task['label']}: {task['completed']}/{task['total'] if task['total'] is not None else '?'} {task['unit']}; approximate ETA {duration(task['eta_seconds'])}"
        )
        if task["details"]:
            lines.append(f"  {json.dumps(task['details'], ensure_ascii=False)}")
    lines.append(
        "Whole-run ETA: 0h 0m 0s (all planned stages complete)."
        if value["state"] == "completed"
        else "Whole-run ETA: unknown until remaining stages have measured runtimes."
    )
    if value["error"]:
        lines.append(f"Error: {value['error']}")
    return "\n".join(lines)
