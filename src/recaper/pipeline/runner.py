"""PipelineRunner — sequentially executes stages."""

from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from recaper.exceptions import StageError
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.progress import ProgressReporter
from recaper.pipeline.stages.base import Stage

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Runs a sequence of stages, reporting progress."""

    def __init__(self, stages: Sequence[Stage], resume: bool = False) -> None:
        self.stages = stages
        self.resume = resume

    async def run(self, ctx: PipelineContext, progress: ProgressReporter) -> None:
        ctx.ensure_dirs()

        for i, stage in enumerate(self.stages, 1):
            if self.resume and stage.is_complete(ctx):
                logger.info("Skipping completed stage: %s", stage.name)
                progress.on_stage_complete(f"{stage.name} (cached)")
                continue

            logger.info("Starting stage %d/%d: %s", i, len(self.stages), stage.name)
            progress.on_stage_start(stage.name, stage.description)

            try:
                await stage.run(ctx, progress)
            except StageError:
                progress.on_error(stage.name, "stage failed")
                raise
            except Exception as exc:
                progress.on_error(stage.name, str(exc))
                raise StageError(stage.name, str(exc)) from exc

            progress.on_stage_complete(stage.name)

        logger.info("Pipeline complete.")


def run_pipeline_sync(
    stages: Sequence[Stage],
    ctx: PipelineContext,
    progress: ProgressReporter,
    resume: bool = False,
) -> None:
    """Synchronous wrapper for running the pipeline."""
    runner = PipelineRunner(stages, resume=resume)
    asyncio.run(runner.run(ctx, progress))
