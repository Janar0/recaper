"""Application configuration via environment variables and CLI overrides."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RecaperConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RECAPER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter LLM
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    openrouter_model: str = Field(
        default="anthropic/claude-sonnet-4-20250514",
        description="Main model for analysis/script generation",
    )
    ocr_model: str = Field(
        default="google/gemini-2.0-flash-001",
        description="Cheap vision model for OCR and panel fallback detection",
    )
    llm_temperature: float = 0.7
    llm_batch_size: int = Field(default=4, description="Panels per LLM request")
    llm_max_retries: int = 3
    llm_max_image_size: int = Field(
        default=1024, description="Max image dimension (px) sent to LLM (downscale to save tokens)",
    )

    # Panel detection
    panel_detector: str = Field(
        default="mosesb/best-comic-panel-detection",
        description="HuggingFace YOLO model for panel detection",
    )
    panel_confidence: float = Field(default=0.3, description="YOLO confidence threshold")

    # Language
    language: str = Field(default="ru", description="Narration language")

    # Pipeline
    work_dir: Path = Field(default=Path("./work"), description="Working directory")
    min_panel_area_ratio: float = Field(
        default=0.02, description="Min panel area as fraction of page"
    )
    panel_padding: int = Field(default=10, description="Padding px when cropping panels")
    min_panel_importance: int = Field(
        default=4, ge=1, le=10,
        description="Min importance score (1-10) to include a panel in the recap script",
    )
