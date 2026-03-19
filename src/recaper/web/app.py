"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from recaper import __version__

_WEB_DIR = Path(__file__).parent


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(title="recaper", version=__version__)

    # Static files (CSS, JS)
    app.mount("/static", StaticFiles(directory=_WEB_DIR / "static"), name="static")

    # Jinja2 templates
    templates = Jinja2Templates(directory=_WEB_DIR / "templates")
    app.state.templates = templates

    # Register routers
    from recaper.web.routes.api import router as api_router
    from recaper.web.routes.pages import router as pages_router

    app.include_router(api_router, prefix="/api")
    app.include_router(pages_router)

    return app
