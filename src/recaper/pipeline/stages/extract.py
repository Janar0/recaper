"""Stage: Extract panels using YOLO model + manhwa vertical split fallback."""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI

from recaper.models import ContentType, PanelInfo, ReadingOrder
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.progress import ProgressReporter
from recaper.pipeline.stages.base import Stage

logger = logging.getLogger(__name__)

MIN_PANEL_PX = 100  # Minimum panel dimension in pixels

# Lazy-loaded YOLO model (heavy import)
_yolo_model = None


def _get_yolo_model(model_id: str):
    """Load YOLO model from HuggingFace, cached after first call."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    # Download the model weights from HuggingFace
    model_path = hf_hub_download(repo_id=model_id, filename="best.pt")
    _yolo_model = YOLO(model_path)
    logger.info("Loaded YOLO model from %s", model_id)
    return _yolo_model


LLM_PANEL_PROMPT = """\
Определи все отдельные панели (кадры) на этой странице {content_type}.

Для каждой панели укажи координаты в процентах от размера страницы (0-100):
- x: левый край
- y: верхний край
- w: ширина
- h: высота

Ответь строго в JSON:
{{"panels": [{{"x": 0, "y": 0, "w": 50, "h": 33}}]}}

Порядок чтения: {reading_order_hint}.
Если вся страница — одна панель (splash), верни один элемент."""


class ExtractStage(Stage):
    @property
    def name(self) -> str:
        return "extract"

    @property
    def description(self) -> str:
        return "Извлечение панелей"

    def is_complete(self, ctx: PipelineContext) -> bool:
        meta = ctx.panels_dir / "metadata.json"
        return meta.exists()

    async def run(self, ctx: PipelineContext, progress: ProgressReporter) -> None:
        ctx.panels_dir.mkdir(parents=True, exist_ok=True)
        reading_order = _reading_order_for(ctx.content_type)
        global_idx = 0

        # Choose detection strategy based on content type
        use_yolo = ctx.content_type in (ContentType.MANGA, ContentType.MANHUA)

        for page_i, page_path in enumerate(ctx.pages):
            progress.on_stage_progress(
                self.name, page_i + 1, len(ctx.pages),
                f"Страница {page_i + 1}/{len(ctx.pages)}",
            )

            img = cv2.imread(str(page_path))
            if img is None:
                logger.warning("Cannot read page %s, skipping", page_path)
                continue

            page_h, page_w = img.shape[:2]
            page_area = page_h * page_w

            # --- Detection ---
            if use_yolo:
                bboxes = _detect_yolo(str(page_path), ctx.config)
                method = "yolo"
            else:
                # Manhwa: vertical strip → split by horizontal gaps
                bboxes = _detect_manhwa_splits(img, ctx.config)
                method = "vsplit"

            # Fallback: LLM vision if detection gave bad results
            if _needs_fallback(bboxes, page_w, page_h) and ctx.config.openrouter_api_key:
                logger.info("Page %d: %s gave %d panels, trying LLM fallback", page_i + 1, method, len(bboxes))
                llm_bboxes = _detect_panels_llm(ctx, page_path, page_w, page_h, reading_order)
                if llm_bboxes:
                    bboxes = llm_bboxes
                    method = "llm"

            # Final fallback: whole page
            if not bboxes:
                bboxes = [(0, 0, page_w, page_h)]
                method = "fullpage"

            # Splash check
            is_splash = False
            if len(bboxes) == 1:
                x, y, w, h = bboxes[0]
                if (w * h) / page_area > 0.8:
                    is_splash = True

            # Sort by reading order (LLM returns pre-sorted)
            if method != "llm":
                bboxes = _sort_panels(bboxes, reading_order)

            for panel_j, (x, y, w, h) in enumerate(bboxes):
                pad = ctx.config.panel_padding
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(page_w, x + w + pad)
                y2 = min(page_h, y + h + pad)

                crop = img[y1:y2, x1:x2]
                crop = _autocrop(crop)

                ch, cw = crop.shape[:2]
                if ch < MIN_PANEL_PX or cw < MIN_PANEL_PX:
                    continue

                is_text = _is_text_only(crop)

                panel_id = f"p{page_i + 1:03d}_{panel_j + 1:03d}"
                out_path = ctx.panels_dir / f"{panel_id}.png"
                cv2.imwrite(str(out_path), crop)

                panel = PanelInfo(
                    panel_id=panel_id,
                    page_index=page_i,
                    panel_index=panel_j,
                    reading_order=global_idx,
                    path=out_path,
                    bbox=(x, y, w, h),
                    is_splash=is_splash and len(bboxes) == 1,
                    is_text_only=is_text,
                )
                ctx.panels.append(panel)
                global_idx += 1

            logger.debug("Page %d: %d panels (%s)", page_i + 1, len(bboxes), method)

        # Save metadata
        meta_path = ctx.panels_dir / "metadata.json"
        meta = [p.model_dump(mode="json") for p in ctx.panels]
        for m in meta:
            m["path"] = str(m["path"])
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info("Extracted %d panels from %d pages", len(ctx.panels), len(ctx.pages))


# ---------------------------------------------------------------------------
# YOLO-based detection (manga/manhua)
# ---------------------------------------------------------------------------

def _detect_yolo(image_path: str, config) -> list[tuple[int, int, int, int]]:
    """Detect panels using YOLOv8 model from HuggingFace."""
    try:
        model = _get_yolo_model(config.panel_detector)
    except Exception as exc:
        logger.warning("Failed to load YOLO model: %s, falling back", exc)
        return []

    results = model(image_path, conf=config.panel_confidence, verbose=False)

    bboxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            w = x2 - x1
            h = y2 - y1
            if w > 0 and h > 0:
                bboxes.append((int(x1), int(y1), int(w), int(h)))

    return bboxes


# ---------------------------------------------------------------------------
# Manhwa vertical split detection
# ---------------------------------------------------------------------------

def _detect_manhwa_splits(
    img: np.ndarray, config, gap_threshold: int = 15, min_gap_height: int = 10
) -> list[tuple[int, int, int, int]]:
    """Split a vertical manhwa strip by detecting horizontal white/empty gaps.

    Manhwa pages are tall vertical strips. Panels are separated by horizontal
    white (or near-white) bands. We scan row-by-row to find these gaps.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # For each row, compute mean brightness
    row_means = np.mean(gray, axis=1)

    # Rows that are "empty" (white gap between panels)
    is_gap = row_means > (255 - gap_threshold)

    # Find contiguous gap regions
    gap_starts = []
    gap_ends = []
    in_gap = False

    for y in range(h):
        if is_gap[y] and not in_gap:
            in_gap = True
            gap_starts.append(y)
        elif not is_gap[y] and in_gap:
            in_gap = False
            gap_ends.append(y)
    if in_gap:
        gap_ends.append(h)

    # Filter: only keep gaps wider than min_gap_height
    split_points = [0]
    for gs, ge in zip(gap_starts, gap_ends):
        if (ge - gs) >= min_gap_height:
            mid = (gs + ge) // 2
            split_points.append(mid)
    split_points.append(h)

    # Create bboxes from split regions
    bboxes = []
    min_area = int(h * w * config.min_panel_area_ratio)
    for i in range(len(split_points) - 1):
        y1 = split_points[i]
        y2 = split_points[i + 1]
        panel_h = y2 - y1
        if panel_h * w >= min_area and panel_h >= MIN_PANEL_PX:
            bboxes.append((0, y1, w, panel_h))

    return bboxes


# ---------------------------------------------------------------------------
# LLM vision fallback (Gemini Flash — cheap)
# ---------------------------------------------------------------------------

def _detect_panels_llm(
    ctx: PipelineContext,
    page_path: Path,
    page_w: int,
    page_h: int,
    reading_order: ReadingOrder,
) -> list[tuple[int, int, int, int]]:
    """Use cheap LLM vision (Gemini Flash) to detect panels."""
    config = ctx.config
    reading_hint = {
        ReadingOrder.RTL: "справа-налево, сверху-вниз",
        ReadingOrder.TOP_DOWN: "сверху-вниз",
        ReadingOrder.LTR: "слева-направо, сверху-вниз",
    }[reading_order]

    prompt = LLM_PANEL_PROMPT.format(
        content_type=ctx.content_type.value,
        reading_order_hint=reading_hint,
    )

    # Downscale image to save tokens
    b64 = _downscale_and_encode(page_path, config.llm_max_image_size)

    client = OpenAI(
        api_key=config.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=config.ocr_model,  # Gemini Flash — cheap
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)

            bboxes = []
            for p in data.get("panels", []):
                x = int(p["x"] / 100 * page_w)
                y = int(p["y"] / 100 * page_h)
                w = int(p["w"] / 100 * page_w)
                h = int(p["h"] / 100 * page_h)
                if w > 0 and h > 0:
                    bboxes.append((x, y, w, h))

            if bboxes:
                logger.info("LLM (Gemini Flash) detected %d panels", len(bboxes))
                return bboxes
        except Exception as exc:
            logger.warning("LLM panel detection attempt %d failed: %s", attempt + 1, exc)
            if attempt == 0:
                time.sleep(2)

    return []


def _needs_fallback(
    bboxes: list[tuple[int, int, int, int]], page_w: int, page_h: int
) -> bool:
    """Decide if primary detection gave suspicious results."""
    if not bboxes:
        return True

    page_area = page_w * page_h

    # Too many tiny boxes — probably noise
    if len(bboxes) > 20:
        return True

    # Boxes cover very little of the page
    total_box_area = sum(w * h for _, _, w, h in bboxes)
    if total_box_area / page_area < 0.25:
        return True

    return False


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _downscale_and_encode(image_path: Path, max_size: int) -> str:
    """Downscale image to max_size on longest side, encode as JPEG base64.

    This saves significant tokens when sending to LLM.
    """
    from PIL import Image
    import io

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size

        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _reading_order_for(content_type: ContentType) -> ReadingOrder:
    if content_type == ContentType.MANGA:
        return ReadingOrder.RTL
    elif content_type == ContentType.MANHWA:
        return ReadingOrder.TOP_DOWN
    return ReadingOrder.LTR


def _sort_panels(
    bboxes: list[tuple[int, int, int, int]], order: ReadingOrder
) -> list[tuple[int, int, int, int]]:
    """Sort panels by reading order."""
    if not bboxes:
        return bboxes

    if order == ReadingOrder.TOP_DOWN:
        return sorted(bboxes, key=lambda b: (b[1], b[0]))

    # Row-based sorting for RTL and LTR
    avg_h = np.mean([h for _, _, _, h in bboxes])
    row_threshold = avg_h * 0.4
    reverse = order == ReadingOrder.RTL

    sorted_by_y = sorted(bboxes, key=lambda b: b[1])
    rows: list[list[tuple[int, int, int, int]]] = []
    current_row: list[tuple[int, int, int, int]] = [sorted_by_y[0]]

    for box in sorted_by_y[1:]:
        if abs(box[1] - current_row[0][1]) < row_threshold:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    rows.append(current_row)

    result = []
    for row in rows:
        result.extend(sorted(row, key=lambda b: b[0], reverse=reverse))
    return result


def _autocrop(img: np.ndarray, threshold: int = 15) -> np.ndarray:
    """Remove white/black borders from a panel image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    edge_mean = np.mean([
        np.mean(gray[0, :]), np.mean(gray[-1, :]),
        np.mean(gray[:, 0]), np.mean(gray[:, -1]),
    ])

    if edge_mean > 200:
        mask = gray < (255 - threshold)
    elif edge_mean < 30:
        mask = gray > threshold
    else:
        return img

    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None:
        return img

    x, y, cw, ch = cv2.boundingRect(coords)
    if cw < w * 0.5 or ch < h * 0.5:
        return img

    return img[y:y + ch, x:x + cw]


def _is_text_only(img: np.ndarray, white_threshold: float = 0.9) -> bool:
    """Check if a panel is mostly white/black (text-only, no artwork)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_pixels = np.sum(gray > 240) + np.sum(gray < 15)
    total = gray.shape[0] * gray.shape[1]
    return (white_pixels / total) > white_threshold
