"""Job management — run pipeline tasks and track their status."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from recaper.config import RecaperConfig
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.progress import ProgressReporter
from recaper.pipeline.runner import PipelineRunner
from recaper.pipeline.stages.analyze import AnalyzeStage
from recaper.pipeline.stages.detect import DetectStage
from recaper.pipeline.stages.extract import ExtractStage
from recaper.pipeline.stages.render import RenderStage
from recaper.pipeline.stages.script import ScriptStage
from recaper.pipeline.stages.unpack import UnpackStage
from recaper.pipeline.stages.voiceover import VoiceoverStage

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobEvent:
    """Single progress event for SSE streaming."""
    type: str  # stage_start | progress | stage_complete | error | done
    data: dict[str, Any]


@dataclass
class Job:
    """Represents a single pipeline job."""
    id: str
    source: Path
    title: str
    work_dir: Path
    status: JobStatus = JobStatus.QUEUED
    current_stage: str = ""
    progress: float = 0.0
    error: str = ""
    events: list[JobEvent] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": str(self.source),
            "title": self.title,
            "work_dir": str(self.work_dir),
            "status": self.status.value,
            "current_stage": self.current_stage,
            "progress": self.progress,
            "error": self.error,
            "result": self.result,
        }


class WebProgressReporter:
    """Progress reporter that appends events to a Job for SSE streaming."""

    def __init__(self, job: Job, event_queue: asyncio.Queue[JobEvent]) -> None:
        self._job = job
        self._queue = event_queue
        self._stage_index = 0
        self._total_stages = 7

    def on_stage_start(self, stage: str, description: str) -> None:
        self._stage_index += 1
        self._job.current_stage = stage
        self._job.progress = (self._stage_index - 1) / self._total_stages * 100
        evt = JobEvent("stage_start", {
            "stage": stage,
            "description": description,
            "step": self._stage_index,
            "total_steps": self._total_stages,
        })
        self._job.events.append(evt)
        self._queue.put_nowait(evt)

    def on_stage_progress(self, stage: str, current: int, total: int, detail: str = "") -> None:
        base = (self._stage_index - 1) / self._total_stages * 100
        within = (current / max(total, 1)) / self._total_stages * 100
        self._job.progress = base + within
        evt = JobEvent("progress", {
            "stage": stage,
            "current": current,
            "total": total,
            "detail": detail,
            "progress": round(self._job.progress, 1),
        })
        self._job.events.append(evt)
        self._queue.put_nowait(evt)

    def on_stage_complete(self, stage: str) -> None:
        self._job.progress = self._stage_index / self._total_stages * 100
        evt = JobEvent("stage_complete", {
            "stage": stage,
            "progress": round(self._job.progress, 1),
        })
        self._job.events.append(evt)
        self._queue.put_nowait(evt)

    def on_error(self, stage: str, error: str) -> None:
        evt = JobEvent("error", {"stage": stage, "error": error})
        self._job.events.append(evt)
        self._queue.put_nowait(evt)


class JobManager:
    """Singleton-style job store and executor."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._queues: dict[str, asyncio.Queue[JobEvent]] = {}

    @property
    def jobs(self) -> list[Job]:
        return list(self._jobs.values())

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def event_queue(self, job_id: str) -> asyncio.Queue[JobEvent] | None:
        return self._queues.get(job_id)

    def create_job(
        self,
        source: Path,
        title: str = "",
        work_dir: Path | None = None,
        model: str = "",
        resume: bool = False,
    ) -> Job:
        job_id = uuid.uuid4().hex[:12]
        effective_work_dir = work_dir or Path(f"./jobs/{job_id}")
        job = Job(
            id=job_id,
            source=source,
            title=title,
            work_dir=effective_work_dir,
        )
        self._jobs[job_id] = job
        self._queues[job_id] = asyncio.Queue()

        asyncio.create_task(self._run(job, model=model, resume=resume))
        return job

    async def _run(self, job: Job, model: str = "", resume: bool = False) -> None:
        queue = self._queues[job.id]
        try:
            job.status = JobStatus.RUNNING
            config = RecaperConfig(work_dir=job.work_dir)
            if model:
                config.openrouter_model = model

            ctx = PipelineContext(
                config=config,
                source_path=job.source,
                title=job.title,
            )
            stages = [
                UnpackStage(),
                DetectStage(),
                ExtractStage(),
                AnalyzeStage(),
                ScriptStage(),
                VoiceoverStage(),
                RenderStage(),
            ]
            reporter = WebProgressReporter(job, queue)
            runner = PipelineRunner(stages, resume=resume)
            await runner.run(ctx, reporter)

            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.result = {
                "content_type": ctx.content_type.value,
                "pages": len(ctx.pages),
                "panels": len(ctx.panels),
                "scenes": len(ctx.script.scenes) if ctx.script else 0,
                "video": str(ctx.video.output_path) if ctx.video else None,
                "video_duration": ctx.video.duration_sec if ctx.video else None,
            }
            queue.put_nowait(JobEvent("done", job.result))

        except Exception as exc:
            logger.exception("Job %s failed", job.id)
            job.status = JobStatus.FAILED
            job.error = str(exc)
            queue.put_nowait(JobEvent("error", {"error": str(exc)}))


# Global instance
job_manager = JobManager()
