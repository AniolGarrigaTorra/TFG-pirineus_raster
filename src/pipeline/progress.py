from __future__ import annotations

import contextlib
import contextvars
import builtins
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

_ORIGINAL_PRINT = builtins.print


def _now() -> float:
    return time.monotonic()


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def format_bytes(value: float | int | None) -> str:
    if value is None:
        return "?"
    size = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size) < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


@dataclass
class DownloadState:
    started_at: float = field(default_factory=_now)
    last_logged_at: float = 0.0
    last_bucket: int = -1


class ProgressReporter:
    """
    Small terminal/log progress reporter for long raster runs.

    It deliberately avoids external UI dependencies. In an interactive terminal it
    keeps a single status line refreshed with carriage returns. In detached logs
    it prints periodic snapshot lines that work well with tail -f.
    """

    def __init__(
        self,
        *,
        run_name: str,
        total_tasks: int,
        stage_totals: dict[str, int] | None = None,
        stream: Any = None,
    ) -> None:
        self.run_name = run_name
        self.total_tasks = max(1, int(total_tasks))
        self.stage_totals = dict(stage_totals or {})
        self.stage_completed = {stage: 0 for stage in self.stage_totals}
        self.completed_tasks = 0
        self.started_at = _now()
        self.current_stage: str | None = None
        self.current_source: str | None = None
        self.current_detail: str | None = None
        self.current_task_started_at: float | None = None
        self.current_stage_task_label: str | None = None
        self.current_stage_task_name: str | None = None
        self.current_stage_task_done = 0
        self.current_stage_task_total: int | None = None
        self.stream = stream or sys.stdout
        mode = os.environ.get("PIRINEUS_PROGRESS", "auto").lower()
        self.dynamic = bool(
            mode not in {"plain", "log", "0", "false"}
            and hasattr(self.stream, "isatty")
            and self.stream.isatty()
        )
        color_mode = os.environ.get("PIRINEUS_PROGRESS_COLOR", "auto").lower()
        self.use_color = bool(
            color_mode not in {"0", "false", "none", "off"}
            and (color_mode in {"1", "true", "always"} or self.dynamic)
        )
        self._line_len = 0
        self._download_states: dict[str, DownloadState] = {}
        self._plain_status_interval_s = 20.0
        self._last_plain_status_at = 0.0
        self._last_plain_status_percent = -1.0

    @property
    def elapsed(self) -> float:
        return _now() - self.started_at

    def _progress_units(self) -> float:
        current_fraction = 0.0
        if self.current_stage_task_total:
            current_fraction = min(
                1.0,
                max(0.0, self.current_stage_task_done / self.current_stage_task_total),
            )
        return min(float(self.total_tasks), float(self.completed_tasks) + current_fraction)

    def _eta(self) -> float | None:
        progress_units = self._progress_units()
        if progress_units <= 0:
            return None
        avg = self.elapsed / progress_units
        return avg * max(0.0, float(self.total_tasks) - progress_units)

    def _percent(self) -> float:
        return min(100.0, 100.0 * self._progress_units() / float(self.total_tasks))

    def _status_text(self) -> str:
        stage_name = self.current_stage or "pending"
        stage_index = 0
        if self.current_stage:
            try:
                stage_index = list(self.stage_totals).index(self.current_stage) + 1
            except ValueError:
                stage_index = 0

        if self.current_stage_task_total:
            task_label = self.current_stage_task_label or "tasks"
            task_text = (
                f"tasks: {self.current_stage_task_done}/"
                f"{self.current_stage_task_total} {task_label}"
            )
        else:
            done = self.stage_completed.get(self.current_stage or "", 0)
            total = self.stage_totals.get(self.current_stage or "")
            task_text = f"tasks: {done}/{total}" if total else "tasks: --"

        stage_text = (
            f"stage {stage_index}/{len(self.stage_totals)}: {stage_name}"
            if self.stage_totals
            else f"stage: {stage_name}"
        )
        task_name = self.current_stage_task_name or self.current_source or self.current_detail
        task_name_text = f" | {task_name}" if task_name else ""
        return (
            f"[run] {self.run_name} | {self._percent():5.1f}% "
            f"| elapsed time: {format_duration(self.elapsed)} "
            f"| ETA: {format_duration(self._eta())} | {stage_text} | {task_text}"
            f"{task_name_text}"
        )

    def _decorate_status(self, text: str) -> str:
        if not self.use_color:
            return text
        return f"\033[1;30;46m {text} \033[0m"

    def _clear_dynamic_line(self) -> None:
        if not self.dynamic or self._line_len <= 0:
            return
        self.stream.write("\r" + " " * self._line_len + "\r")
        self.stream.flush()
        self._line_len = 0

    def _write_line(self, text: str) -> None:
        self._clear_dynamic_line()
        _ORIGINAL_PRINT(text, file=self.stream, flush=True)

    def _should_render_plain_status(self, *, force: bool) -> bool:
        if force:
            return True
        now = _now()
        percent = self._percent()
        if percent >= 100.0 and self._last_plain_status_percent < 100.0:
            return True
        if percent - self._last_plain_status_percent >= 5.0:
            return True
        return now - self._last_plain_status_at >= self._plain_status_interval_s

    def render_status(self, *, force: bool = False) -> None:
        text = self._status_text()
        rendered = self._decorate_status(text)
        if self.dynamic:
            pad = max(0, self._line_len - len(text))
            self.stream.write("\r" + rendered + " " * pad)
            self.stream.flush()
            self._line_len = len(text)
        else:
            if not self._should_render_plain_status(force=force):
                return
            self._write_line(text)
            self._last_plain_status_at = _now()
            self._last_plain_status_percent = self._percent()

    def start(self) -> None:
        self._write_line("=" * 72)
        self._write_line(f"Pirineus Raster run: {self.run_name}")
        self._write_line(f"Planned tasks: {self.total_tasks}")
        if self.stage_totals:
            stages = ", ".join(
                f"{stage}={count}" for stage, count in self.stage_totals.items()
            )
            self._write_line(f"Stage plan: {stages}")
        self._write_line("=" * 72)
        self.render_status(force=True)

    def log(self, message: str, *, level: str = "info") -> None:
        self._write_line(f"[{level}] {message}")
        if self.dynamic:
            self.render_status()

    def start_task(
        self,
        *,
        stage: str,
        source_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        self.current_stage = stage
        self.current_source = source_id
        self.current_detail = detail
        self.current_task_started_at = _now()
        self.current_stage_task_label = None
        self.current_stage_task_name = None
        self.current_stage_task_done = 0
        self.current_stage_task_total = None

        task_no = self.completed_tasks + 1
        stage_total = self.stage_totals.get(stage)
        stage_done = self.stage_completed.get(stage, 0)
        stage_text = (
            f"stage {stage} task {stage_done + 1}/{stage_total}"
            if stage_total
            else f"stage {stage}"
        )
        source_text = f" | source {source_id}" if source_id else ""
        detail_text = f" | {detail}" if detail else ""
        self._write_line(
            f"[start] task {task_no}/{self.total_tasks} | {stage_text}"
            f"{source_text}{detail_text}"
        )
        self.render_status(force=True)

    def set_stage_task_total(self, total: int, *, label: str = "tasks") -> None:
        self.current_stage_task_total = max(0, int(total))
        self.current_stage_task_done = 0
        self.current_stage_task_label = label
        self.current_stage_task_name = None
        self.render_status(force=True)

    def advance_stage_task(
        self,
        *,
        increment: int = 1,
        name: str | None = None,
    ) -> None:
        if self.current_stage_task_total is None:
            return
        self.current_stage_task_done = min(
            self.current_stage_task_total,
            self.current_stage_task_done + max(0, int(increment)),
        )
        if name:
            self.current_stage_task_name = name
        self.render_status()

    def finish_task(self, *, output_count: int | None = None) -> None:
        duration = (
            _now() - self.current_task_started_at
            if self.current_task_started_at is not None
            else None
        )
        stage = self.current_stage or "task"
        if self.current_stage_task_total is not None:
            self.current_stage_task_done = self.current_stage_task_total
        self.completed_tasks = min(self.total_tasks, self.completed_tasks + 1)
        self.stage_completed[stage] = self.stage_completed.get(stage, 0) + 1

        output_text = "" if output_count is None else f" | outputs {output_count}"
        self._write_line(
            f"[done] {stage}"
            f"{' | source ' + self.current_source if self.current_source else ''}"
            f" | duration {format_duration(duration)}{output_text}"
        )
        self.current_detail = None
        self.current_task_started_at = None
        self.current_stage_task_label = None
        self.current_stage_task_name = None
        self.current_stage_task_done = 0
        self.current_stage_task_total = None
        self.render_status(force=True)

    def finish(self) -> None:
        self.completed_tasks = min(self.completed_tasks, self.total_tasks)
        self._clear_dynamic_line()
        self._write_line("=" * 72)
        self._write_line(
            f"Run finished | elapsed {format_duration(self.elapsed)} "
            f"| tasks {self.completed_tasks}/{self.total_tasks}"
        )
        self._write_line("=" * 72)

    def download_progress(
        self,
        *,
        output_path: Path,
        downloaded: int,
        total: int | None,
        attempt: int | None = None,
        done: bool = False,
    ) -> None:
        key = str(output_path)
        state = self._download_states.setdefault(key, DownloadState())
        now = _now()
        elapsed = max(0.001, now - state.started_at)
        speed = downloaded / elapsed

        should_log = done
        if total and total > 0:
            bucket = int((downloaded / total) * 20)  # 5% buckets
            should_log = should_log or bucket > state.last_bucket
            state.last_bucket = max(state.last_bucket, bucket)
        should_log = should_log or (now - state.last_logged_at >= 10.0)

        if not should_log:
            return

        state.last_logged_at = now
        attempt_text = f" | attempt {attempt}" if attempt is not None else ""

        if total and total > 0:
            percent = min(100.0, 100.0 * downloaded / total)
            remaining = max(0, total - downloaded)
            eta = remaining / speed if speed > 0 and not done else 0
            message = (
                f"[download] {Path(output_path).name} | {percent:5.1f}% "
                f"| {format_bytes(downloaded)}/{format_bytes(total)} "
                f"| {format_bytes(speed)}/s | ETA {format_duration(eta)}"
                f"{attempt_text}"
            )
        else:
            message = (
                f"[download] {Path(output_path).name} | {format_bytes(downloaded)} "
                f"| {format_bytes(speed)}/s{attempt_text}"
            )

        self._write_line(message)
        if self.dynamic:
            self.render_status()


_CURRENT_REPORTER: contextvars.ContextVar[ProgressReporter | None] = (
    contextvars.ContextVar("pirineus_progress_reporter", default=None)
)


def get_progress_reporter() -> ProgressReporter | None:
    return _CURRENT_REPORTER.get()


@contextlib.contextmanager
def use_progress_reporter(reporter: ProgressReporter) -> Iterator[ProgressReporter]:
    token = _CURRENT_REPORTER.set(reporter)
    previous_print = builtins.print

    def _routed_print(*args: Any, **kwargs: Any) -> None:
        active_reporter = get_progress_reporter()
        output_file = kwargs.get("file")
        if active_reporter is not reporter or output_file not in (None, sys.stdout, reporter.stream):
            previous_print(*args, **kwargs)
            return

        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(str(arg) for arg in args)
        if end and end != "\n":
            text += str(end)

        if text == "":
            reporter._write_line("")
            return

        for line in text.splitlines():
            reporter.log(line)

    builtins.print = _routed_print
    try:
        yield reporter
    except Exception:
        reporter._clear_dynamic_line()
        raise
    finally:
        builtins.print = previous_print
        _CURRENT_REPORTER.reset(token)


def progress_log(message: str, *, level: str = "info") -> None:
    reporter = get_progress_reporter()
    if reporter is not None:
        reporter.log(message, level=level)
    else:
        _ORIGINAL_PRINT(f"[{level}] {message}", flush=True)


def progress_download(
    *,
    output_path: Path,
    downloaded: int,
    total: int | None,
    attempt: int | None = None,
    done: bool = False,
) -> None:
    reporter = get_progress_reporter()
    if reporter is None:
        return
    reporter.download_progress(
        output_path=output_path,
        downloaded=downloaded,
        total=total,
        attempt=attempt,
        done=done,
    )


def progress_set_stage_task_total(total: int, *, label: str = "tasks") -> None:
    reporter = get_progress_reporter()
    if reporter is None:
        return
    reporter.set_stage_task_total(total, label=label)


def progress_advance_stage_task(
    *,
    increment: int = 1,
    name: str | None = None,
) -> None:
    reporter = get_progress_reporter()
    if reporter is None:
        return
    reporter.advance_stage_task(increment=increment, name=name)
