"""Stage: Analyze panels via LLM vision using annotated full-page images.

Instead of sending individual panel crops, draws colored bounding boxes on the
full page image and sends ONE low-res annotated image per page. The LLM performs
both review (merge/discard corrections) and analysis (action, characters, etc.)
in a single call. This saves ~75-85% of image tokens compared to per-panel crops.
"""

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

from recaper.exceptions import LLMError
from recaper.models import PanelAnalysis, PanelInfo
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.progress import ProgressReporter
from recaper.pipeline.stages.base import Stage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette for bounding boxes (16 distinct high-contrast colors)
# ---------------------------------------------------------------------------

PANEL_COLORS = [
    (255, 0, 0), (0, 0, 255), (0, 180, 0), (255, 140, 0),
    (200, 0, 200), (0, 200, 200), (255, 255, 0), (0, 255, 128),
    (128, 0, 255), (255, 0, 128), (0, 128, 255), (180, 120, 0),
    (128, 128, 0), (0, 128, 128), (128, 0, 128), (255, 128, 128),
]

COLOR_NAMES_RU = [
    "красный", "синий", "зелёный", "оранжевый",
    "фиолетовый", "голубой", "жёлтый", "мятный",
    "индиго", "розовый", "васильковый", "коричневый",
    "оливковый", "бирюзовый", "пурпурный", "лососевый",
]

# ---------------------------------------------------------------------------
# Prompt template — combined review + analysis
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """\
Ты анализируешь {content_type} "{title}".
На изображении — страница {page_num}. Панели обведены цветными рамками:
{panel_list}

{context_block}

{characters_block}

ЗАДАЧА 1 — Проверь нарезку:
- Если панели ошибочно разрезаны (одна сцена на 2-3 куска) → укажи merge
- Если панель пустая, декоративная или мусор → укажи discard
- Если всё ок — ничего не указывай в corrections

ЗАДАЧА 2 — Для каждой оставшейся панели (не discard, после merge) опиши:
1. Что происходит (действие, эмоции персонажей)
2. Какие персонажи присутствуют — используй ТОЛЬКО имена из реестра выше, если персонаж уже известен. \
Для НОВОГО персонажа дай имя (или прозвище по внешности) и подробное описание внешности: \
цвет волос, причёска, цвет глаз, одежда, отличительные черты.
3. Диалоги (если есть текст — переведи на русский)
4. Звуковые эффекты (SFX)
5. Настроение / атмосфера
6. Важность для сюжета (1–10)
7. Брак (is_defective): true если панель нечитаема или бесполезна

Ответь строго в JSON (без markdown-обёртки):
{{
  "corrections": [
    {{"action": "merge", "panels": ["p001_002", "p001_003"], "reason": "одна сцена разрезана"}},
    {{"action": "discard", "panel": "p001_005", "reason": "пустая рамка"}}
  ],
  "panels": [
    {{
      "panel_id": "id панели",
      "action": "описание действия",
      "characters": ["персонаж1", "персонаж2"],
      "dialogue": [{{"speaker": "имя", "text": "оригинал", "translated": "перевод"}}],
      "sfx": ["звук1"],
      "mood": "настроение",
      "visual_notes": "визуальные приёмы",
      "importance": 7,
      "is_defective": false
    }}
  ],
  "new_characters": {{
    "Имя персонажа": "подробное описание внешности: цвет волос, причёска, глаза, одежда, особенности"
  }},
  "scene_summary": "краткое описание сцены",
  "narrative_beat": "exposition / rising_action / climax / falling_action / resolution"
}}"""


# ---------------------------------------------------------------------------
# Annotated page image builder
# ---------------------------------------------------------------------------

def _get_font(size: int = 14):
    """Try to load a decent font, fall back to PIL default."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


def _build_annotated_page(
    page_path: Path,
    panels: list[PanelInfo],
    max_size: int = 512,
    max_aspect: float = 5.0,
) -> list[tuple[bytes, list[PanelInfo]]]:
    """Draw colored bounding boxes on a page image and return JPEG chunks.

    For normal pages: returns a single (jpeg_bytes, panels) tuple.
    For tall manhwa pages (aspect > max_aspect): splits into vertical chunks,
    each with its own subset of panels.
    """
    img = Image.open(page_path).convert("RGB")
    w, h = img.size
    aspect = h / w if w > 0 else 1.0

    if aspect > max_aspect and len(panels) > 1:
        return _build_chunked_page(img, panels, max_size, max_aspect)

    # Single page — draw all boxes and downscale
    annotated = _draw_boxes(img, panels)
    jpeg_bytes = _downscale_to_jpeg(annotated, max_size)
    return [(jpeg_bytes, panels)]


def _build_chunked_page(
    img: Image.Image,
    panels: list[PanelInfo],
    max_size: int,
    max_aspect: float,
) -> list[tuple[bytes, list[PanelInfo]]]:
    """Split a tall page into vertical chunks, each within max_aspect ratio."""
    w, h = img.size
    chunk_height = int(w * max_aspect)
    n_chunks = math.ceil(h / chunk_height)

    chunks: list[tuple[bytes, list[PanelInfo]]] = []

    for ci in range(n_chunks):
        y_start = ci * chunk_height
        y_end = min((ci + 1) * chunk_height, h)

        # Find panels whose bbox center falls in this chunk
        chunk_panels = []
        for p in panels:
            _, py, _, ph = p.bbox
            center_y = py + ph / 2
            if y_start <= center_y < y_end:
                chunk_panels.append(p)

        if not chunk_panels:
            continue

        # Crop the chunk from the page
        chunk_img = img.crop((0, y_start, w, y_end))

        # Draw boxes with adjusted y-coordinates
        draw = ImageDraw.Draw(chunk_img)
        font = _get_font(max(10, w // 40))

        for panel in chunk_panels:
            idx = panels.index(panel)
            color = PANEL_COLORS[idx % len(PANEL_COLORS)]
            px, py, pw, ph = panel.bbox
            # Adjust y relative to chunk
            adj_y = py - y_start
            box = (px, adj_y, px + pw, adj_y + ph)
            for offset in range(3):  # 3px thick border
                draw.rectangle(
                    (box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset),
                    outline=color,
                )
            # Label
            label = panel.panel_id
            _draw_label(draw, font, label, (px, adj_y), color)

        jpeg_bytes = _downscale_to_jpeg(chunk_img, max_size)
        chunks.append((jpeg_bytes, chunk_panels))

    return chunks


def _draw_boxes(img: Image.Image, panels: list[PanelInfo]) -> Image.Image:
    """Draw colored bounding boxes and labels on a copy of the image."""
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    w, _ = img.size
    font = _get_font(max(10, w // 40))

    for i, panel in enumerate(panels):
        color = PANEL_COLORS[i % len(PANEL_COLORS)]
        px, py, pw, ph = panel.bbox
        box = (px, py, px + pw, py + ph)
        for offset in range(3):  # 3px thick border
            draw.rectangle(
                (box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset),
                outline=color,
            )
        # Label
        _draw_label(draw, font, panel.panel_id, (px, py), color)

    return annotated


def _draw_label(
    draw: ImageDraw.ImageDraw, font, text: str, pos: tuple[int, int], color: tuple
) -> None:
    """Draw a text label with colored background at the given position."""
    x, y = pos
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 2
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=color,
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def _downscale_to_jpeg(img: Image.Image, max_size: int) -> bytes:
    """Downscale image to max_size on longest side and encode as JPEG."""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Panel merge helper (ported from review.py)
# ---------------------------------------------------------------------------

def _merge_panels(src: PanelInfo, dst: PanelInfo) -> None:
    """Merge src panel image into dst panel by vertical concatenation."""
    img_dst = cv2.imread(str(dst.path))
    img_src = cv2.imread(str(src.path))
    if img_dst is None or img_src is None:
        return

    target_w = max(img_dst.shape[1], img_src.shape[1])
    if img_dst.shape[1] != target_w:
        scale = target_w / img_dst.shape[1]
        img_dst = cv2.resize(img_dst, (target_w, int(img_dst.shape[0] * scale)))
    if img_src.shape[1] != target_w:
        scale = target_w / img_src.shape[1]
        img_src = cv2.resize(img_src, (target_w, int(img_src.shape[0] * scale)))

    if src.reading_order < dst.reading_order:
        merged = np.vstack([img_src, img_dst])
    else:
        merged = np.vstack([img_dst, img_src])

    cv2.imwrite(str(dst.path), merged, [cv2.IMWRITE_JPEG_QUALITY, 95])

    x1 = min(src.bbox[0], dst.bbox[0])
    y1 = min(src.bbox[1], dst.bbox[1])
    x2 = max(src.bbox[0] + src.bbox[2], dst.bbox[0] + dst.bbox[2])
    y2 = max(src.bbox[1] + src.bbox[3], dst.bbox[1] + dst.bbox[3])
    dst.bbox = (x1, y1, x2 - x1, y2 - y1)


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class AnalyzeStage(Stage):
    @property
    def name(self) -> str:
        return "analyze"

    @property
    def description(self) -> str:
        return "LLM-анализ панелей"

    def is_complete(self, ctx: PipelineContext) -> bool:
        summary = ctx.analysis_dir / "summary.json"
        return summary.exists()

    def restore(self, ctx: PipelineContext) -> None:
        summary_path = ctx.analysis_dir / "summary.json"
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        ctx.analyses = [PanelAnalysis(**d) for d in data["analyses"]]
        logger.info("Restored %d analyses from %s", len(ctx.analyses), summary_path)

    async def run(self, ctx: PipelineContext, progress: ProgressReporter) -> None:
        cfg = ctx.config

        if not cfg.openrouter_api_key:
            raise LLMError("RECAPER_OPENROUTER_API_KEY is not set")

        # Filter out text-only panels
        panels = [p for p in ctx.panels if not p.is_text_only]
        if not panels:
            logger.warning("No panels to analyze")
            return

        client = OpenAI(
            api_key=cfg.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        # Group panels by page
        pages: dict[int, list[PanelInfo]] = {}
        for p in panels:
            pages.setdefault(p.page_index, []).append(p)

        previous_summary = ""
        character_registry: dict[str, str] = {}
        all_analyses: list[PanelAnalysis] = []
        all_discarded: set[str] = set()
        all_merged: dict[str, str] = {}

        ctx.analysis_dir.mkdir(parents=True, exist_ok=True)

        total_pages = len(pages)
        for page_i, (page_idx, page_panels) in enumerate(sorted(pages.items())):
            progress.on_stage_progress(
                self.name, page_i + 1, total_pages,
                f"Страница {page_idx + 1} ({len(page_panels)} панелей)",
            )

            # Get page image path
            if page_idx >= len(ctx.pages):
                logger.warning("Page index %d out of range, skipping", page_idx)
                continue
            page_path = ctx.pages[page_idx]

            # Build annotated page chunks
            chunks = _build_annotated_page(
                page_path, page_panels,
                max_size=cfg.llm_annotated_image_size,
                max_aspect=cfg.analyze_max_aspect_ratio,
            )

            for chunk_i, (jpeg_bytes, chunk_panels) in enumerate(chunks):
                # Build panel list for prompt
                panel_list_parts = []
                for i, panel in enumerate(chunk_panels):
                    color_name = COLOR_NAMES_RU[page_panels.index(panel) % len(COLOR_NAMES_RU)]
                    panel_list_parts.append(f"{panel.panel_id} ({color_name})")
                panel_list = ", ".join(panel_list_parts)

                context_block = (
                    f"Контекст предыдущих панелей: {previous_summary}"
                    if previous_summary
                    else "Это начало главы."
                )

                if character_registry:
                    chars_list = "\n".join(
                        f"- {name}: {desc}" for name, desc in character_registry.items()
                    )
                    characters_block = f"Известные персонажи (используй эти имена!):\n{chars_list}"
                else:
                    characters_block = "Персонажи пока не встречались. Опиши внешность каждого нового персонажа подробно."

                prompt = ANALYSIS_PROMPT.format(
                    content_type=ctx.content_type.value,
                    title=ctx.title or "Без названия",
                    page_num=page_idx + 1,
                    panel_list=panel_list,
                    context_block=context_block,
                    characters_block=characters_block,
                )

                # Build message with single annotated image
                b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
                messages = [
                    {"role": "system", "content": "Ты — эксперт по анализу манги и манхвы. Отвечай только на русском языке."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ],
                    },
                ]

                # Call LLM
                result = self._call_with_retry(client, cfg, messages)

                # Apply corrections (merge/discard)
                corrections = result.get("corrections", [])
                discarded, merged = self._apply_corrections(corrections, chunk_panels, ctx)
                all_discarded.update(discarded)
                all_merged.update(merged)

                # Parse analyses for remaining panels
                remaining = [p for p in chunk_panels if p.panel_id not in discarded]
                batch_analyses = self._parse_response(result, remaining)
                all_analyses.extend(batch_analyses)

                # Update character registry
                new_chars = result.get("new_characters", {})
                if isinstance(new_chars, dict):
                    for name, desc in new_chars.items():
                        if name and desc and name not in character_registry:
                            character_registry[name] = desc
                            logger.debug("New character registered: %s", name)

                # Update summary for next page context
                if "scene_summary" in result:
                    previous_summary += " " + result.get("scene_summary", "")
                    previous_summary = previous_summary.strip()

                # Save batch result
                batch_name = f"page_{page_idx + 1:03d}"
                if len(chunks) > 1:
                    batch_name += f"_chunk_{chunk_i + 1}"
                batch_path = ctx.analysis_dir / f"{batch_name}.json"
                batch_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )

        # Remove discarded panels from ctx.panels
        if all_discarded:
            before = len(ctx.panels)
            ctx.panels = [p for p in ctx.panels if p.panel_id not in all_discarded]
            for i, p in enumerate(ctx.panels):
                p.reading_order = i
            logger.info("Corrections: removed %d panels (%d → %d)", before - len(ctx.panels), before, len(ctx.panels))

            # Re-save panel metadata
            meta_path = ctx.panels_dir / "metadata.json"
            meta = [p.model_dump(mode="json") for p in ctx.panels]
            for m in meta:
                m["path"] = str(m["path"])
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        ctx.analyses = all_analyses

        # Save summary
        summary_path = ctx.analysis_dir / "summary.json"
        summary_data = {
            "total_panels": len(panels),
            "total_pages": total_pages,
            "corrections": {
                "discarded": list(all_discarded),
                "merged": all_merged,
            },
            "summary": previous_summary,
            "character_registry": character_registry,
            "analyses": [a.model_dump() for a in all_analyses],
        }
        summary_path.write_text(
            json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info("Analyzed %d panels across %d pages", len(all_analyses), total_pages)

    def _call_with_retry(self, client: OpenAI, cfg, messages: list[dict]) -> dict:
        last_error = None
        for attempt in range(cfg.llm_max_retries):
            try:
                response = client.chat.completions.create(
                    model=cfg.openrouter_model,
                    messages=messages,
                    temperature=cfg.llm_temperature,
                    max_tokens=4096,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except json.JSONDecodeError as exc:
                raw = response.choices[0].message.content or ""
                logger.warning("Failed to parse JSON on attempt %d: %s", attempt + 1, exc)
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.split("\n")
                    cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    last_error = exc
            except Exception as exc:
                last_error = exc
                logger.warning("LLM call attempt %d failed: %s", attempt + 1, exc)

            if attempt < cfg.llm_max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.info("Retrying in %ds...", wait)
                time.sleep(wait)

        raise LLMError(f"LLM call failed after {cfg.llm_max_retries} attempts: {last_error}")

    def _apply_corrections(
        self,
        corrections: list[dict],
        panels: list[PanelInfo],
        ctx: PipelineContext,
    ) -> tuple[set[str], dict[str, str]]:
        """Apply merge/discard corrections from LLM response.

        Returns (discarded_ids, merged: src_id → dst_id).
        """
        discarded: set[str] = set()
        merged: dict[str, str] = {}
        panel_map = {p.panel_id: p for p in panels}

        for correction in corrections:
            action = correction.get("action", "")

            if action == "discard":
                panel_id = correction.get("panel", "")
                if panel_id in panel_map:
                    discarded.add(panel_id)
                    logger.info("Correction: discard %s (%s)", panel_id, correction.get("reason", ""))

            elif action == "merge":
                merge_ids = correction.get("panels", [])
                if len(merge_ids) >= 2:
                    # Merge all into the first panel
                    dst_id = merge_ids[0]
                    dst = panel_map.get(dst_id)
                    if not dst:
                        continue
                    for src_id in merge_ids[1:]:
                        src = panel_map.get(src_id)
                        if src:
                            _merge_panels(src, dst)
                            discarded.add(src_id)
                            merged[src_id] = dst_id
                            logger.info(
                                "Correction: merge %s → %s (%s)",
                                src_id, dst_id, correction.get("reason", ""),
                            )

        return discarded, merged

    def _parse_response(self, result: dict, panels: list[PanelInfo]) -> list[PanelAnalysis]:
        analyses = []
        raw_panels = result.get("panels", [])

        # Build a lookup by panel_id from LLM response
        raw_by_id = {}
        for raw in raw_panels:
            pid = raw.get("panel_id", "")
            if pid:
                raw_by_id[pid] = raw

        for panel in panels:
            raw = raw_by_id.get(panel.panel_id)
            if raw is None:
                # Fallback: try positional matching
                idx = panels.index(panel)
                if idx < len(raw_panels):
                    raw = raw_panels[idx]

            if raw:
                analysis = PanelAnalysis(
                    panel_id=panel.panel_id,
                    action=raw.get("action", ""),
                    characters=raw.get("characters", []),
                    dialogue=raw.get("dialogue", []),
                    sfx=raw.get("sfx", []),
                    mood=raw.get("mood", ""),
                    visual_notes=raw.get("visual_notes", ""),
                    importance=min(10, max(1, raw.get("importance", 5))),
                    is_defective=raw.get("is_defective", False),
                )
            else:
                analysis = PanelAnalysis(panel_id=panel.panel_id)
            analyses.append(analysis)

        return analyses
