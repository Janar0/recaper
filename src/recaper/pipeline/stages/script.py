"""Stage: Generate a cohesive narrative script from panel analyses."""

from __future__ import annotations

import json
import logging
import time

from openai import OpenAI

from recaper.exceptions import LLMError
from recaper.models import NarrativeScript, SceneBlock
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.progress import ProgressReporter
from recaper.pipeline.stages.base import Stage

logger = logging.getLogger(__name__)

SCRIPT_PROMPT = """\
Ты — профессиональный рассказчик и сценарист. На основе анализа панелей {content_type} "{title}" \
создай захватывающий нарративный пересказ на русском языке.

Правила:
- НЕ описывай каждую панель по отдельности. Объединяй их в связные сцены.
- Используй литературный русский язык, метафоры, сравнения.
- Передавай эмоции персонажей через нарратив.
- Вставляй ключевые реплики персонажей в прямую речь (переведённые на русский).
- Делай паузы в нужных местах для драматического эффекта.
- Каждый блок нарратива привязан к 1–5 панелям.
- Длительность одного блока: 10–30 секунд озвучки (примерно 30–90 слов).
- Общая длительность: примерно 1 минута на 5–8 страниц.

Вот анализ панелей:
{analyses_json}

Ответь строго в JSON (без markdown-обёртки):
{{
  "title": "Название эпизода на русском",
  "scenes": [
    {{
      "scene_id": 1,
      "narration": "Текст нарратива для озвучки...",
      "panel_ids": ["p001_001", "p001_002"],
      "mood": "tense",
      "pacing": "slow",
      "transition": "crossfade"
    }}
  ]
}}

Доступные mood: calm, tense, dramatic, action, sad, romantic, comedic, mysterious, neutral
Доступные pacing: slow, normal, fast
Доступные transition: crossfade, wipe_left, dissolve, black, slide_left"""


class ScriptStage(Stage):
    @property
    def name(self) -> str:
        return "script"

    @property
    def description(self) -> str:
        return "Генерация сценария"

    def is_complete(self, ctx: PipelineContext) -> bool:
        return ctx.script_path.exists()

    async def run(self, ctx: PipelineContext, progress: ProgressReporter) -> None:
        cfg = ctx.config

        if not cfg.openrouter_api_key:
            raise LLMError("RECAPER_OPENROUTER_API_KEY is not set")

        if not ctx.analyses:
            logger.warning("No analyses available, cannot generate script")
            return

        progress.on_stage_progress(self.name, 0, 1, "Генерация сценария...")

        # Filter out low-importance panels
        threshold = cfg.min_panel_importance
        scored = [a for a in ctx.analyses if a.importance >= threshold]
        skipped = len(ctx.analyses) - len(scored)
        if skipped:
            logger.info("Skipping %d panels with importance < %d", skipped, threshold)
        if not scored:
            logger.warning("All panels below importance threshold %d, using all", threshold)
            scored = ctx.analyses

        # Prepare analyses for the prompt
        analyses_data = []
        for a in scored:
            analyses_data.append({
                "panel_id": a.panel_id,
                "action": a.action,
                "characters": a.characters,
                "dialogue": a.dialogue,
                "mood": a.mood,
                "importance": a.importance,
            })

        prompt = SCRIPT_PROMPT.format(
            content_type=ctx.content_type.value,
            title=ctx.title or "Без названия",
            analyses_json=json.dumps(analyses_data, ensure_ascii=False, indent=2),
        )

        client = OpenAI(
            api_key=cfg.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        result = self._call_with_retry(client, cfg, prompt)

        # Parse into NarrativeScript
        scenes = []
        for raw_scene in result.get("scenes", []):
            scene = SceneBlock(
                scene_id=raw_scene.get("scene_id", len(scenes) + 1),
                narration=raw_scene.get("narration", ""),
                panel_ids=raw_scene.get("panel_ids", []),
                mood=raw_scene.get("mood", "neutral"),
                pacing=raw_scene.get("pacing", "normal"),
                transition=raw_scene.get("transition", "crossfade"),
            )
            scenes.append(scene)

        script = NarrativeScript(
            title=result.get("title", ctx.title or "Без названия"),
            content_type=ctx.content_type,
            scenes=scenes,
            total_panels=len(ctx.panels),
        )

        ctx.script = script

        # Save to file
        ctx.script_path.write_text(
            script.model_dump_json(indent=2), encoding="utf-8"
        )

        progress.on_stage_progress(self.name, 1, 1, "Готово")
        logger.info("Generated script with %d scenes", len(scenes))

    def _call_with_retry(self, client: OpenAI, cfg, prompt: str) -> dict:
        last_error = None
        for attempt in range(cfg.llm_max_retries):
            try:
                response = client.chat.completions.create(
                    model=cfg.openrouter_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Ты — профессиональный сценарист для видео-рекапов манги. Отвечай только на русском языке. Отвечай строго в JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=cfg.llm_temperature,
                    max_tokens=8192,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except json.JSONDecodeError as exc:
                raw = response.choices[0].message.content or ""
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
                logger.warning("Script generation attempt %d failed: %s", attempt + 1, exc)

            if attempt < cfg.llm_max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.info("Retrying in %ds...", wait)
                time.sleep(wait)

        raise LLMError(f"Script generation failed after {cfg.llm_max_retries} attempts: {last_error}")
