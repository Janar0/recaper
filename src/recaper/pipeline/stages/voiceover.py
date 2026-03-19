"""Stage: Generate TTS audio for each scene using Qwen3-TTS."""

from __future__ import annotations

import logging
import wave
from pathlib import Path

from recaper.exceptions import TTSError
from recaper.models import AudioSegment
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.progress import ProgressReporter
from recaper.pipeline.stages.base import Stage

logger = logging.getLogger(__name__)


def _wav_duration(path: Path) -> float:
    """Return duration in seconds for a WAV file."""
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


class VoiceoverStage(Stage):
    @property
    def name(self) -> str:
        return "voiceover"

    @property
    def description(self) -> str:
        return "Озвучка сцен (TTS)"

    def is_complete(self, ctx: PipelineContext) -> bool:
        if not ctx.script:
            return False
        audio_dir = ctx.audio_dir
        return all(
            (audio_dir / f"scene_{s.scene_id:03d}.wav").exists()
            for s in ctx.script.scenes
        )

    async def run(self, ctx: PipelineContext, progress: ProgressReporter) -> None:
        if not ctx.script or not ctx.script.scenes:
            logger.warning("No script available, skipping voiceover")
            return

        cfg = ctx.config
        scenes = ctx.script.scenes
        total = len(scenes)

        progress.on_stage_progress(self.name, 0, total, "Загрузка TTS модели...")

        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise TTSError(
                "qwen-tts not installed. Run: pip install 'recaper[tts]'"
            ) from exc

        # Select device
        if torch.cuda.is_available():
            device = "cuda:0"
            dtype = torch.bfloat16
        else:
            device = "cpu"
            dtype = torch.float32
            logger.warning("CUDA not available, using CPU for TTS (will be slow)")

        try:
            model = Qwen3TTSModel.from_pretrained(
                cfg.tts_model,
                device_map=device,
                dtype=dtype,
            )
        except Exception as exc:
            raise TTSError(f"Failed to load TTS model '{cfg.tts_model}': {exc}") from exc

        audio_dir = ctx.audio_dir
        audio_dir.mkdir(parents=True, exist_ok=True)

        segments: list[AudioSegment] = []

        for i, scene in enumerate(scenes):
            progress.on_stage_progress(
                self.name, i, total,
                f"Озвучка сцены {i + 1}/{total}...",
            )

            out_path = audio_dir / f"scene_{scene.scene_id:03d}.wav"

            # Skip if already generated (for resume)
            if out_path.exists():
                duration = _wav_duration(out_path)
                segments.append(AudioSegment(
                    scene_id=scene.scene_id,
                    audio_path=out_path,
                    duration_sec=duration,
                ))
                logger.info("Scene %d already voiced (%.1fs)", scene.scene_id, duration)
                continue

            text = scene.narration.strip()
            if not text:
                logger.warning("Scene %d has empty narration, skipping", scene.scene_id)
                continue

            try:
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=cfg.tts_language,
                    speaker=cfg.tts_speaker,
                )
            except Exception as exc:
                raise TTSError(
                    f"TTS failed for scene {scene.scene_id}: {exc}"
                ) from exc

            # Save WAV
            import soundfile as sf

            sf.write(str(out_path), wavs[0], sr)

            duration = _wav_duration(out_path)
            segments.append(AudioSegment(
                scene_id=scene.scene_id,
                audio_path=out_path,
                duration_sec=duration,
            ))

            logger.info(
                "Scene %d voiced: %.1fs (%d chars)",
                scene.scene_id, duration, len(text),
            )

        ctx.audio_segments = segments

        progress.on_stage_progress(self.name, total, total, "Готово")

        total_dur = sum(s.duration_sec for s in segments)
        logger.info(
            "Voiceover complete: %d segments, total %.1fs",
            len(segments), total_dur,
        )
