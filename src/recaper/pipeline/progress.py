"""Progress reporting protocol and Rich console implementation."""

from __future__ import annotations

from typing import Protocol

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


class ProgressReporter(Protocol):
    """Protocol for reporting pipeline progress."""

    def on_stage_start(self, stage: str, description: str) -> None: ...

    def on_stage_progress(self, stage: str, current: int, total: int, detail: str = "") -> None: ...

    def on_stage_complete(self, stage: str) -> None: ...

    def on_error(self, stage: str, error: str) -> None: ...


class RichProgressReporter:
    """Reports progress to the terminal using Rich."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._progress: Progress | None = None
        self._task_id: int | None = None

    def on_stage_start(self, stage: str, description: str) -> None:
        if self._progress is not None:
            self._progress.stop()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        )
        self._task_id = self._progress.add_task(description, total=None)
        self._progress.start()

    def on_stage_progress(self, stage: str, current: int, total: int, detail: str = "") -> None:
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=current, total=total)
            if detail:
                self._progress.update(self._task_id, description=detail)

    def on_stage_complete(self, stage: str) -> None:
        if self._progress is not None:
            if self._task_id is not None:
                self._progress.update(self._task_id, completed=self._progress.tasks[self._task_id].total or 1, total=self._progress.tasks[self._task_id].total or 1)
            self._progress.stop()
            self._progress = None
            self._task_id = None
        self.console.print(f"  [green]✓[/green] {stage}")

    def on_error(self, stage: str, error: str) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
        self.console.print(f"  [red]✗[/red] {stage}: {error}")


class SilentReporter:
    """No-op reporter for testing."""

    def on_stage_start(self, stage: str, description: str) -> None:
        pass

    def on_stage_progress(self, stage: str, current: int, total: int, detail: str = "") -> None:
        pass

    def on_stage_complete(self, stage: str) -> None:
        pass

    def on_error(self, stage: str, error: str) -> None:
        pass
