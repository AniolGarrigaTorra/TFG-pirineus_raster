from __future__ import annotations

import builtins
import io
import unittest
from pathlib import Path

from src.pipeline.progress import (
    ProgressReporter,
    progress_advance_stage_task,
    progress_download,
    progress_set_stage_task_total,
    use_progress_reporter,
)


class ProgressReporterTests(unittest.TestCase):
    def test_plain_prints_are_routed_and_restored(self):
        original_print = builtins.print
        stream = io.StringIO()
        reporter = ProgressReporter(
            run_name="progress_smoke",
            total_tasks=1,
            stage_totals={"download": 1},
            stream=stream,
        )

        with use_progress_reporter(reporter):
            print("source message")
            self.assertIsNot(builtins.print, original_print)

        self.assertIs(builtins.print, original_print)
        self.assertIn("[info] source message", stream.getvalue())

    def test_download_progress_logs_percentages(self):
        stream = io.StringIO()
        reporter = ProgressReporter(
            run_name="download_smoke",
            total_tasks=1,
            stage_totals={"download": 1},
            stream=stream,
        )

        with use_progress_reporter(reporter):
            progress_download(
                output_path=Path("example.tif"),
                downloaded=50,
                total=100,
                attempt=1,
            )

        output = stream.getvalue()
        self.assertIn("[download] example.tif", output)
        self.assertIn("50.0%", output)

    def test_stage_subtasks_advance_status_percent(self):
        stream = io.StringIO()
        reporter = ProgressReporter(
            run_name="subtask_smoke",
            total_tasks=2,
            stage_totals={"download": 1, "clip": 1},
            stream=stream,
        )

        with use_progress_reporter(reporter):
            reporter.start_task(stage="download", source_id="source_a")
            progress_set_stage_task_total(4, label="downloads")
            progress_advance_stage_task(name="tile_01.tif")

        output = stream.getvalue()
        self.assertIn("12.5%", output)
        self.assertIn("stage 1/2: download", output)
        self.assertIn("tasks: 1/4 downloads", output)
        self.assertIn("tile_01.tif", output)


if __name__ == "__main__":
    unittest.main()
