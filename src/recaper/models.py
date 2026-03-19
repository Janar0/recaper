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


class SceneBlock(BaseModel):
    """A narrative scene covering 1-5 panels."""

    scene_id: int
    narration: str                   # Text for TTS (Russian)
    panel_ids: list[str]             # Panel IDs to show during narration
    mood: str = "neutral"            # For music/animation selection
    pacing: str = "normal"           # slow / normal / fast
    transition: str = "crossfade"    # Transition to next scene


class NarrativeScript(BaseModel):
    """Full narrative script for a chapter."""

    title: str
    content_type: ContentType
    scenes: list[SceneBlock]
    total_panels: int
