"""PipelineContext — mutable state flowing through all stages."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from recaper.config import RecaperConfig
from recaper.models import ContentType, NarrativeScript, PanelAnalysis, PanelInfo


@dataclass
class PipelineContext:
    """Holds all intermediate data produced by pipeline stages."""

    config: RecaperConfig
    source_path: Path
    title: str = ""
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # Populated by stages
    pages: list[Path] = field(default_factory=list)
    content_type: ContentType = ContentType.MANGA
    panels: list[PanelInfo] = field(default_factory=list)
    analyses: list[PanelAnalysis] = field(default_factory=list)
    script: NarrativeScript | None = None

    @property
    def work_dir(self) -> Path:
        return self.config.work_dir

    @property
    def pages_dir(self) -> Path:
        return self.work_dir / "pages"

    @property
    def panels_dir(self) -> Path:
        return self.work_dir / "panels"

    @property
    def analysis_dir(self) -> Path:
        return self.work_dir / "analysis"

    @property
    def script_path(self) -> Path:
        return self.work_dir / "script.json"

    def ensure_dirs(self) -> None:
        """Create all working subdirectories."""
        for d in (self.pages_dir, self.panels_dir, self.analysis_dir):
            d.mkdir(parents=True, exist_ok=True)
