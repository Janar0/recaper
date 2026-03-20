"""HTML page routes — server-side rendered pages via Jinja2."""

from __future__ import annotations

from fastapi import APIRouter, Request

from recaper import __version__
from recaper.web.services.jobs import JobStatus, job_manager

router = APIRouter(tags=["pages"])


def _templates(request: Request):
    return request.app.state.templates


@router.get("/")
async def index(request: Request):
    """Main dashboard — job list and new job form."""
    jobs_list = list(reversed(job_manager.jobs))
    stats = {
        "total": len(jobs_list),
        "running": sum(1 for j in jobs_list if j.status == JobStatus.RUNNING),
        "completed": sum(1 for j in jobs_list if j.status == JobStatus.COMPLETED),
        "failed": sum(1 for j in jobs_list if j.status == JobStatus.FAILED),
    }
    return _templates(request).TemplateResponse("index.html", {
        "request": request,
        "version": __version__,
        "jobs": jobs_list,
        "stats": stats,
        "current_page": "jobs",
    })


@router.get("/jobs/{job_id}/view")
async def job_detail(request: Request, job_id: str):
    """Job detail page with real-time progress."""
    job = job_manager.get(job_id)
    if not job:
        return _templates(request).TemplateResponse("404.html", {
            "request": request,
            "message": "Задание не найдено",
        }, status_code=404)
    return _templates(request).TemplateResponse("job.html", {
        "request": request,
        "version": __version__,
        "job": job,
        "current_page": "job_detail",
    })


@router.get("/config/view")
async def config_page(request: Request):
    """Configuration overview page."""
    from recaper.config import RecaperConfig
    cfg = RecaperConfig()
    return _templates(request).TemplateResponse("config.html", {
        "request": request,
        "version": __version__,
        "config": cfg,
        "current_page": "config",
    })
