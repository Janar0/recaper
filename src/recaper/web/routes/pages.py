"""HTML page routes — server-side rendered pages via Jinja2."""

from __future__ import annotations

from fastapi import APIRouter, Request

from recaper import __version__
from recaper.web.services.jobs import job_manager

router = APIRouter(tags=["pages"])


def _templates(request: Request):
    return request.app.state.templates


@router.get("/")
async def index(request: Request):
    """Main dashboard — job list and new job form."""
    return _templates(request).TemplateResponse("index.html", {
        "request": request,
        "version": __version__,
        "jobs": list(reversed(job_manager.jobs)),
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
    })
