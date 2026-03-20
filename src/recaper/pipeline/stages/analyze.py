"""Stage: Analyze panels via LLM vision (OpenRouter, OpenAI-compatible API)."""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path

from openai import OpenAI

from recaper.exceptions import LLMError
from recaper.models import PanelAnalysis
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.progress import ProgressReporter
from recaper.pipeline.stages.base import Stage

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """\
Ты анализируешь {content_type} "{title}".
Вот панели {start_idx}–{end_idx} из текущей главы.

{context_block}

{characters_block}

Для каждой панели опиши на русском языке:
1. Что происходит (действие, эмоции персонажей)
2. Какие персонажи присутствуют — используй ТОЛЬКО имена из реестра выше, если персонаж уже известен. \
Для НОВОГО персонажа дай имя (или прозвище по внешности) и подробное описание внешности: \
цвет волос, причёска, цвет глаз, одежда, отличительные черты.
3. Диалоги (если есть текст — переведи на русский)
4. Звуковые эффекты (SFX)
5. Настроение / атмосфера
6. Важность для сюжета (1–10)
7. Брак (is_defective): true если панель пустая, только рамка/белое пространство, слишком мала или нечитаема

Ответь строго в JSON (без markdown-обёртки):
{{
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


def _image_to_base64(path: Path, max_size: int = 0) -> str:
    """Encode image as base64, optionally downscaling to save tokens."""
    if max_size > 0:
        import io
        from PIL import Image

        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


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

        batch_size = cfg.llm_batch_size
        batches = [panels[i:i + batch_size] for i in range(0, len(panels), batch_size)]
        previous_summary = ""
        # Character registry: name → appearance description (accumulated across batches)
        character_registry: dict[str, str] = {}
        all_analyses: list[PanelAnalysis] = []

        ctx.analysis_dir.mkdir(parents=True, exist_ok=True)

        for batch_i, batch in enumerate(batches):
            progress.on_stage_progress(
                self.name, batch_i + 1, len(batches),
                f"Батч {batch_i + 1}/{len(batches)} ({len(batch)} панелей)",
            )

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
                start_idx=batch[0].reading_order + 1,
                end_idx=batch[-1].reading_order + 1,
                context_block=context_block,
                characters_block=characters_block,
            )

            # Build vision messages with panel images
            image_content = []
            max_img = cfg.llm_max_image_size
            for panel in batch:
                b64 = _image_to_base64(panel.path, max_size=max_img)
                mime = "image/jpeg" if max_img > 0 else "image/png"
                image_content.append({
                    "type": "text",
                    "text": f"Панель {panel.panel_id} (#{panel.reading_order + 1}):",
                })
                image_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })

            messages = [
                {"role": "system", "content": "Ты — эксперт по анализу манги и манхвы. Отвечай только на русском языке."},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}] + image_content,
                },
            ]

            # Call LLM with retries
            result = self._call_with_retry(client, cfg, messages)

            # Parse response
            batch_analyses = self._parse_response(result, batch)
            all_analyses.extend(batch_analyses)

            # Update character registry with new characters from this batch
            new_chars = result.get("new_characters", {})
            if isinstance(new_chars, dict):
                for name, desc in new_chars.items():
                    if name and desc and name not in character_registry:
                        character_registry[name] = desc
                        logger.debug("New character registered: %s", name)

            # Extract summary for next batch context
            if "scene_summary" in result:
                previous_summary += " " + result.get("scene_summary", "")
                previous_summary = previous_summary.strip()

            # Save batch result
            batch_path = ctx.analysis_dir / f"batch_{batch_i + 1:03d}.json"
            batch_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        ctx.analyses = all_analyses

        # Save summary
        summary_path = ctx.analysis_dir / "summary.json"
        summary_data = {
            "total_panels": len(panels),
            "total_batches": len(batches),
            "summary": previous_summary,
            "character_registry": character_registry,
            "analyses": [a.model_dump() for a in all_analyses],
        }
        summary_path.write_text(
            json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info("Analyzed %d panels in %d batches", len(all_analyses), len(batches))

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
                # Try to extract JSON from response
                raw = response.choices[0].message.content or ""
                logger.warning("Failed to parse JSON on attempt %d: %s", attempt + 1, exc)
                # Try stripping markdown code fences
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

    def _parse_response(self, result: dict, batch: list) -> list[PanelAnalysis]:
        analyses = []
        raw_panels = result.get("panels", [])

        for i, panel in enumerate(batch):
            if i < len(raw_panels):
                raw = raw_panels[i]
                analysis = PanelAnalysis(
                    panel_id=panel.panel_id,
                    action=raw.get("action", ""),
                    characters=raw.get("characters", []),
                    dialogue=raw.get("dialogue", []),
                    sfx=raw.get("sfx", []),
                    mood=raw.get("mood", ""),
                    visual_notes=raw.get("visual_notes", ""),
                    importance=min(10, max(1, raw.get("importance", 5))),
                )
            else:
                analysis = PanelAnalysis(panel_id=panel.panel_id)
            analyses.append(analysis)

        return analyses
