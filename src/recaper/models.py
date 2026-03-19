"""Shared data models used across the pipeline."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    MANGA = "manga"      # Japanese, RTL reading, B&W
    MANHWA = "manhwa"    # Korean, vertical scroll, color
    MANHUA = "manhua"    # Chinese, varies


class ReadingOrder(str, Enum):
    RTL = "rtl"              # Right-to-left (manga)
    TOP_DOWN = "top_down"    # Top-to-bottom (manhwa)
    LTR = "ltr"              # Left-to-right (manhua/western)


class PanelInfo(BaseModel):
    """Metadata for a single extracted panel."""

    panel_id: str                     # e.g. "p001_002" (page 1, panel 2)
    page_index: int                   # Source page number (0-based)
    panel_index: int                  # Panel index on the page (0-based)
    reading_order: int                # Global reading order across all pages
    path: Path                        # Path to the extracted panel image
    bbox: tuple[int, int, int, int]   # (x, y, w, h) on the source page
    is_splash: bool = False           # Full-page splash panel
    is_text_only: bool = False        # Mostly text, no artwork


class PanelAnalysis(BaseModel):
    """LLM analysis result for a single panel."""

    panel_id: str
    action: str = ""
    characters: list[str] = Field(default_factory=list)
    dialogue: list[dict] = Field(default_factory=list)
    sfx: list[str] = Field(default_factory=list)
    mood: str = ""
    visual_notes: str = ""
    importance: int = Field(default=5, ge=1, le=10)
    is_defective: bool = False  # True if panel has no visual content worth showing


class PanelNarration(BaseModel):
    """Narration text for a single panel."""

    panel_id: str
    text: str  # TTS narration text for this panel


class SceneBlock(BaseModel):
    """A narrative scene covering 1+ panels, each with its own narration."""

    scene_id: int
    panel_narrations: list[PanelNarration] = Field(default_factory=list)
    narration: str = ""              # Fallback full text (computed or legacy)
    panel_ids: list[str] = Field(default_factory=list)  # Fallback (legacy)
    mood: str = "neutral"
    pacing: str = "normal"
    transition: str = "crossfade"

    def effective_panel_ids(self) -> list[str]:
        if self.panel_narrations:
            return [pn.panel_id for pn in self.panel_narrations]
        return self.panel_ids


class NarrativeScript(BaseModel):
    """Full narrative script for a chapter."""

    title: str
    content_type: ContentType
    scenes: list[SceneBlock]
    total_panels: int


class AudioSegment(BaseModel):
    """TTS output for a single panel narration (or whole scene as fallback)."""

    scene_id: int
    panel_id: str = ""   # Empty = scene-level audio (legacy fallback)
    audio_path: Path
    duration_sec: float


class VideoMeta(BaseModel):
    """Metadata for the rendered video."""

    output_path: Path
    duration_sec: float
    resolution: tuple[int, int]
    scenes_count: int
