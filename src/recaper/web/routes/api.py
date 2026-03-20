"""REST API endpoints for job management and pipeline control."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from recaper.config import RecaperConfig
from recaper.web.services.jobs import JobStatus, job_manager

router = APIRouter(tags=["api"])


# --- Request / Response schemas ---


class JobCreateRequest(BaseModel):
    source: str = Field(..., description="Path to .cbz/.cbr file or image directory")
    title: str = ""
    work_dir: str = ""
    model: str = ""
    resume: bool = False


class ConfigResponse(BaseModel):
    openrouter_model: str
    ocr_model: str
    language: str
    llm_batch_size: int
    llm_temperature: float
    panel_confidence: float
    min_panel_importance: int
    video_fps: int
    video_width: int
    video_height: int
    ken_burns_zoom: float
    transition_duration: float
    tts_model: str
    tts_speaker: str
    tts_language: str


# --- Jobs ---


@router.get("/jobs/stats")
async def job_stats():
    """Aggregated job counts for the dashboard."""
    jobs = job_manager.jobs
    return {
        "total": len(jobs),
        "running": sum(1 for j in jobs if j.status == JobStatus.RUNNING),
        "completed": sum(1 for j in jobs if j.status == JobStatus.COMPLETED),
        "failed": sum(1 for j in jobs if j.status == JobStatus.FAILED),
        "queued": sum(1 for j in jobs if j.status == JobStatus.QUEUED),
    }


@router.get("/jobs")
async def list_jobs():
    """List all jobs (newest first)."""
    return [j.to_dict() for j in reversed(job_manager.jobs)]


@router.post("/jobs", status_code=201)
async def create_job(req: JobCreateRequest):
    """Create and start a new pipeline job."""
    source = Path(req.source)
    if not source.exists():
        raise HTTPException(404, f"Source not found: {req.source}")

    work_dir = Path(req.work_dir) if req.work_dir else None
    job = job_manager.create_job(
        source=source,
        title=req.title,
        work_dir=work_dir,
        model=req.model,
        resume=req.resume,
    )
    return job.to_dict()


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status and details."""
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str):
    """SSE stream of real-time job progress events."""
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    queue = job_manager.event_queue(job_id)
    if not queue:
        raise HTTPException(404, "Event queue not available")

    async def generate():
        # Send past events first (replay)
        for evt in job.events:
            yield f"event: {evt.type}\ndata: {json.dumps(evt.data, ensure_ascii=False)}\n\n"

        # Stream new events
        while True:
            try:
                evt = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield f"event: {evt.type}\ndata: {json.dumps(evt.data, ensure_ascii=False)}\n\n"
                if evt.type in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# --- Files ---


@router.get("/jobs/{job_id}/files")
async def list_job_files(job_id: str):
    """List output files for a completed job."""
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    work = job.work_dir
    if not work.exists():
        return {"files": []}

    files = []
    for f in sorted(work.rglob("*")):
        if f.is_file():
            rel = f.relative_to(work)
            files.append({
                "name": str(rel),
                "size": f.stat().st_size,
                "type": f.suffix.lstrip("."),
            })
    return {"files": files}


# --- Config ---


@router.get("/config")
async def get_config():
    """Return current configuration (from env / defaults)."""
    cfg = RecaperConfig()
    return ConfigResponse(
        openrouter_model=cfg.openrouter_model,
        ocr_model=cfg.ocr_model,
        language=cfg.language,
        llm_batch_size=cfg.llm_batch_size,
        llm_temperature=cfg.llm_temperature,
        panel_confidence=cfg.panel_confidence,
        min_panel_importance=cfg.min_panel_importance,
        video_fps=cfg.video_fps,
        video_width=cfg.video_width,
        video_height=cfg.video_height,
        ken_burns_zoom=cfg.ken_burns_zoom,
        transition_duration=cfg.transition_duration,
        tts_model=cfg.tts_model,
        tts_speaker=cfg.tts_speaker,
        tts_language=cfg.tts_language,
    )
