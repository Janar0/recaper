"""Stage: Auto-detect content type (manga / manhwa / manhua)."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from recaper.models import ContentType
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.progress import ProgressReporter
from recaper.pipeline.stages.base import Stage

logger = logging.getLogger(__name__)

# Thresholds
SATURATION_THRESHOLD = 30       # Mean saturation above this → color (manhwa/manhua)
ASPECT_RATIO_THRESHOLD = 3.0    # Height/width > this → vertical scroll (manhwa)
SAMPLE_PAGES = 5                # Number of pages to sample for detection


class DetectStage(Stage):
    @property
    def name(self) -> str:
        return "detect"

    @property
    def description(self) -> str:
        return "Определение типа контента"

    async def run(self, ctx: PipelineContext, progress: ProgressReporter) -> None:
        pages = ctx.pages
        if not pages:
            logger.warning("No pages to analyze, defaulting to manga")
            ctx.content_type = ContentType.MANGA
            return

        sample = pages[:SAMPLE_PAGES]
        saturations: list[float] = []
        aspect_ratios: list[float] = []

        for i, page_path in enumerate(sample):
            progress.on_stage_progress(self.name, i + 1, len(sample), f"Анализ страницы {i + 1}")
            img = cv2.imread(str(page_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            aspect_ratios.append(h / w)

            # Mean saturation in HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mean_sat = float(np.mean(hsv[:, :, 1]))
            saturations.append(mean_sat)

        avg_saturation = np.mean(saturations) if saturations else 0
        avg_aspect = np.mean(aspect_ratios) if aspect_ratios else 1.5
        is_color = avg_saturation > SATURATION_THRESHOLD
        is_tall = avg_aspect > ASPECT_RATIO_THRESHOLD

        if is_tall:
            content_type = ContentType.MANHWA
        elif is_color:
            content_type = ContentType.MANHUA
        else:
            content_type = ContentType.MANGA

        ctx.content_type = content_type
        logger.info(
            "Detected: %s (avg_saturation=%.1f, avg_aspect=%.2f, is_color=%s, is_tall=%s)",
            content_type.value, avg_saturation, avg_aspect, is_color, is_tall,
        )
