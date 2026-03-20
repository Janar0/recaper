"""Stage: Generate a cohesive narrative script from panel analyses."""

from __future__ import annotations

import json
import logging
import time

from openai import OpenAI

from recaper.exceptions import LLMError
from recaper.models import NarrativeScript, PanelNarration, SceneBlock
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.progress import ProgressReporter
from recaper.pipeline.stages.base import Stage

logger = logging.getLogger(__name__)

SCRIPT_PROMPT = """\
Ты — закадровый голос для видео-пересказа {content_type} "{title}".

Правила стиля текста:
- Пиши так, как будто спокойно рассказываешь другу что происходит.
- Короткие простые предложения. Настоящее время.
- ЗАПРЕЩЕНО: восклицательные знаки, многоточия, пафосные слова ("невероятно", "поражает", "мощь", "эпичный", "внезапно", "шокирующий", "удивительно").
- ЗАПРЕЩЕНО: вопросительные конструкции для нагнетания ("что же будет?", "неужели?", "как такое возможно?").
- Никакой театральности — это обычный пересказ, не драматический спектакль.
- Ровный тон на протяжении всего текста. Даже экшен-сцены описывай спокойно и по факту.
- Каждой панели — одна реплика (8–20 слов), описывающая что конкретно видно.
- Диалоги: короткая цитата без лишних слов вокруг.

Вот анализ панелей (id=panel_id, a=action, c=characters, m=mood, i=importance):
{analyses_json}

Ответь строго в JSON (без markdown-обёртки):
{{
  "title": "Название эпизода на русском",
  "scenes": [
    {{
      "scene_id": 1,
      "panel_narrations": [
        {{"panel_id": "p001_001", "text": "Короткая реплика для этой панели"}},
        {{"panel_id": "p001_002", "text": "Реплика для следующей панели"}}
      ],
      "mood": "tense",
      "pacing": "normal",
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

    def restore(self, ctx: PipelineContext) -> None:
        data = json.loads(ctx.script_path.read_text(encoding="utf-8"))
        ctx.script = NarrativeScript(**data)
        logger.info("Restored script with %d scenes from %s", len(ctx.script.scenes), ctx.script_path)

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

        # Prepare compact analyses for the prompt (shorter keys, truncated action,
        # no dialogue — it's already reflected in action text). Saves ~40-50% tokens.
        analyses_data = []
        for a in scored:
            analyses_data.append({
                "id": a.panel_id,
                "a": a.action[:100] if a.action else "",
                "c": a.characters,
                "m": a.mood,
                "i": a.importance,
            })

        prompt = SCRIPT_PROMPT.format(
            content_type=ctx.content_type.value,
            title=ctx.title or "Без названия",
            analyses_json=json.dumps(analyses_data, ensure_ascii=False, indent=1),
        )

        client = OpenAI(
            api_key=cfg.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        result = self._call_with_retry(client, cfg, prompt)

        # Parse into NarrativeScript
        scenes = []
        for raw_scene in result.get("scenes", []):
            raw_pn = raw_scene.get("panel_narrations", [])
            panel_narrations = [
                PanelNarration(panel_id=p["panel_id"], text=p["text"])
                for p in raw_pn if p.get("panel_id") and p.get("text")
            ]
            # Derive fallback fields from panel_narrations if present
            narration = raw_scene.get("narration") or " ".join(pn.text for pn in panel_narrations)
            panel_ids = raw_scene.get("panel_ids") or [pn.panel_id for pn in panel_narrations]
            scene = SceneBlock(
                scene_id=raw_scene.get("scene_id", len(scenes) + 1),
                panel_narrations=panel_narrations,
                narration=narration,
                panel_ids=panel_ids,
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
                            "content": "Ты — спокойный закадровый комментатор. Пиши простым разговорным языком без пафоса. Без восклицаний и театральности. Отвечай строго в JSON.",
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
