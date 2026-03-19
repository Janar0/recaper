"""Abstract base class for pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recaper.pipeline.context import PipelineContext
    from recaper.pipeline.progress import ProgressReporter


class Stage(ABC):
    """A single step in the processing pipeline."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this stage (e.g. 'unpack', 'extract')."""

    @property
    def description(self) -> str:
        """Human-readable description shown in progress output."""
        return self.name

    @abstractmethod
    async def run(self, ctx: PipelineContext, progress: ProgressReporter) -> None:
        """Execute the stage, mutating *ctx* with results.

        Raise ``StageError`` on failure.
        """

    def is_complete(self, ctx: PipelineContext) -> bool:
        """Return True if this stage's output already exists (for --resume)."""
        return False
