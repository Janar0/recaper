"""Stage: LLM-based review of extracted panels — merge, discard, or keep."""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

from recaper.models import PanelInfo
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.progress import ProgressReporter
from recaper.pipeline.stages.base import Stage

logger = logging.getLogger(__name__)

REVIEW_PROMPT = """\
Ты проверяешь нарезку панелей для {content_type} "{title}".

На изображении — все панели со страницы {page_num}, пронумерованы. Каждая помечена номером.

Реши для каждой панели:
- "keep" — панель нормальная, оставить
- "discard" — панель пустая, декоративная, дублирует другую или не несёт смысла (рамка, фон, мусор)
- "merge" — эти панели должны быть одной (укажи с какой объединить)

Правила:
- Панели с диалогами и персонажами — всегда keep
- Панели с только текстом без картинки — discard
- Если две панели — это явно одна сцена, разрезанная пополам — merge
- Сплэш-панели (на всю страницу) — keep
- Маленькие полоски без содержания — discard

Ответь строго в JSON (без markdown-обёртки):
{{
  "decisions": [
    {{"panel_idx": 0, "action": "keep"}},
    {{"panel_idx": 1, "action": "discard", "reason": "пустая рамка"}},
    {{"panel_idx": 2, "action": "merge", "merge_with": 3, "reason": "одна сцена разрезана"}}
  ]
}}"""


def _build_contact_sheet(
    panels: list[PanelInfo], max_size: int = 1024, jpeg_quality: int = 85
) -> bytes:
    """Build a labeled contact sheet image from panel images.

    Returns JPEG bytes with numbered panel thumbnails arranged in a grid.
    """
    images = []
    for p in panels:
        img = Image.open(p.path).convert("RGB")
        images.append(img)

    n = len(images)
    if n == 0:
        return b""

    cols = min(n, max(2, math.isqrt(n)))
    rows = math.ceil(n / cols)

    # Target cell size
    cell_w = max_size // cols
    cell_h = cell_w  # square cells

    # Resize all to fit
    thumbs = []
    for img in images:
        img.thumbnail((cell_w - 4, cell_h - 20), Image.LANCZOS)
        thumbs.append(img)

    sheet_w = cols * cell_w
    sheet_h = rows * cell_h
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, thumb in enumerate(thumbs):
        col = i % cols
        row = i // cols
        x = col * cell_w + 2
        y = row * cell_h + 16
        sheet.paste(thumb, (x, y))
        # Draw label
        draw.text((x, row * cell_h), f"#{i}", fill=(255, 0, 0), font=font)

    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue()


class ReviewStage(Stage):
    @property
    def name(self) -> str:
        return "review"

    @property
    def description(self) -> str:
        return "LLM-ревью нарезки панелей"

    def is_complete(self, ctx: PipelineContext) -> bool:
        return (ctx.panels_dir / "review_done.json").exists()

    def restore(self, ctx: PipelineContext) -> None:
        """Re-apply review results: reload filtered panel metadata."""
        meta_path = ctx.panels_dir / "metadata.json"
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        ctx.panels = [PanelInfo(**d) for d in data]
        logger.info("Restored %d reviewed panels from %s", len(ctx.panels), meta_path)

    async def run(self, ctx: PipelineContext, progress: ProgressReporter) -> None:
        cfg = ctx.config
        if not cfg.openrouter_api_key:
            logger.info("No API key, skipping panel review")
            return

        if not ctx.panels:
            return

        client = OpenAI(
            api_key=cfg.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        # Group panels by page
        pages: dict[int, list[PanelInfo]] = {}
        for p in ctx.panels:
            pages.setdefault(p.page_index, []).append(p)

        panels_to_discard: set[str] = set()
        panels_to_merge: dict[str, str] = {}  # panel_id → merge_into panel_id

        total_pages = len(pages)
        for i, (page_idx, page_panels) in enumerate(sorted(pages.items())):
            progress.on_stage_progress(
                self.name, i + 1, total_pages,
                f"Ревью страницы {page_idx + 1}",
            )

            # Skip pages with 1 panel (nothing to review)
            if len(page_panels) <= 1:
                continue

            # Build contact sheet
            sheet_bytes = _build_contact_sheet(
                page_panels, max_size=cfg.llm_max_image_size,
                jpeg_quality=cfg.contact_sheet_quality,
            )
            if not sheet_bytes:
                continue

            b64 = base64.b64encode(sheet_bytes).decode("utf-8")

            prompt = REVIEW_PROMPT.format(
                content_type=ctx.content_type.value,
                title=ctx.title or "Без названия",
                page_num=page_idx + 1,
            )

            # Call LLM
            result = self._call_review(client, cfg, prompt, b64)
            if not result:
                continue

            decisions = result.get("decisions", [])
            for dec in decisions:
                idx = dec.get("panel_idx", -1)
                if idx < 0 or idx >= len(page_panels):
                    continue
                action = dec.get("action", "keep")
                panel = page_panels[idx]

                if action == "discard":
                    panels_to_discard.add(panel.panel_id)
                    logger.info(
                        "Review: discard %s (%s)",
                        panel.panel_id, dec.get("reason", ""),
                    )
                elif action == "merge":
                    merge_idx = dec.get("merge_with", -1)
                    if 0 <= merge_idx < len(page_panels) and merge_idx != idx:
                        target = page_panels[merge_idx]
                        panels_to_merge[panel.panel_id] = target.panel_id
                        logger.info(
                            "Review: merge %s → %s (%s)",
                            panel.panel_id, target.panel_id, dec.get("reason", ""),
                        )

        # Apply merge: create merged panel images
        for src_id, dst_id in panels_to_merge.items():
            src_panel = next((p for p in ctx.panels if p.panel_id == src_id), None)
            dst_panel = next((p for p in ctx.panels if p.panel_id == dst_id), None)
            if src_panel and dst_panel:
                self._merge_panels(src_panel, dst_panel)
                panels_to_discard.add(src_id)

        # Remove discarded panels
        before = len(ctx.panels)
        ctx.panels = [p for p in ctx.panels if p.panel_id not in panels_to_discard]

        # Re-number reading order
        for i, p in enumerate(ctx.panels):
            p.reading_order = i

        # Re-save metadata
        meta_path = ctx.panels_dir / "metadata.json"
        meta = [p.model_dump(mode="json") for p in ctx.panels]
        for m in meta:
            m["path"] = str(m["path"])
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # Mark as complete
        review_path = ctx.panels_dir / "review_done.json"
        review_data = {
            "discarded": list(panels_to_discard),
            "merged": panels_to_merge,
            "before": before,
            "after": len(ctx.panels),
        }
        review_path.write_text(json.dumps(review_data, ensure_ascii=False, indent=2), encoding="utf-8")

        removed = before - len(ctx.panels)
        if removed:
            logger.info("Review complete: removed %d panels (%d → %d)", removed, before, len(ctx.panels))
        else:
            logger.info("Review complete: all %d panels kept", len(ctx.panels))

        progress.on_stage_progress(self.name, total_pages, total_pages, "Готово")

    def _call_review(self, client: OpenAI, cfg, prompt: str, b64: str) -> dict | None:
        for attempt in range(cfg.llm_max_retries):
            try:
                response = client.chat.completions.create(
                    model=cfg.llm_fallback_model or cfg.ocr_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ],
                    }],
                    temperature=0.1,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                # Strip markdown fences if present
                if "```" in content:
                    lines = content.split("\n")
                    content = "\n".join(l for l in lines if not l.strip().startswith("```"))
                return json.loads(content)
            except Exception as exc:
                logger.warning("Review LLM call attempt %d failed: %s", attempt + 1, exc)
                if attempt < cfg.llm_max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
        return None

    @staticmethod
    def _merge_panels(src: PanelInfo, dst: PanelInfo) -> None:
        """Merge src panel image into dst panel by vertical concatenation."""
        img_dst = cv2.imread(str(dst.path))
        img_src = cv2.imread(str(src.path))
        if img_dst is None or img_src is None:
            return

        # Resize to same width
        target_w = max(img_dst.shape[1], img_src.shape[1])
        if img_dst.shape[1] != target_w:
            scale = target_w / img_dst.shape[1]
            img_dst = cv2.resize(img_dst, (target_w, int(img_dst.shape[0] * scale)))
        if img_src.shape[1] != target_w:
            scale = target_w / img_src.shape[1]
            img_src = cv2.resize(img_src, (target_w, int(img_src.shape[0] * scale)))

        # Stack: determine order by reading_order
        if src.reading_order < dst.reading_order:
            merged = np.vstack([img_src, img_dst])
        else:
            merged = np.vstack([img_dst, img_src])

        cv2.imwrite(str(dst.path), merged, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Update dst bbox to cover both
        x1 = min(src.bbox[0], dst.bbox[0])
        y1 = min(src.bbox[1], dst.bbox[1])
        x2 = max(src.bbox[0] + src.bbox[2], dst.bbox[0] + dst.bbox[2])
        y2 = max(src.bbox[1] + src.bbox[3], dst.bbox[1] + dst.bbox[3])
        dst.bbox = (x1, y1, x2 - x1, y2 - y1)
