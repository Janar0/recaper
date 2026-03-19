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

    def _expected_audio_paths(self, ctx) -> list:
        """Return expected (path, scene_id, panel_id) tuples based on script."""
        result = []
        if not ctx.script:
            return result
        for scene in ctx.script.scenes:
            if scene.panel_narrations:
                for pn in scene.panel_narrations:
                    path = ctx.audio_dir / f"panel_{pn.panel_id}.wav"
                    result.append((path, scene.scene_id, pn.panel_id))
            else:
                path = ctx.audio_dir / f"scene_{scene.scene_id:03d}.wav"
                result.append((path, scene.scene_id, ""))
        return result

    def is_complete(self, ctx: PipelineContext) -> bool:
        if not ctx.script:
            return False
        return all(p.exists() for p, _, _ in self._expected_audio_paths(ctx))

    def restore(self, ctx: PipelineContext) -> None:
        from recaper.models import AudioSegment
        segments = []
        for path, scene_id, panel_id in self._expected_audio_paths(ctx):
            if path.exists():
                duration = _wav_duration(path)
                segments.append(AudioSegment(
                    scene_id=scene_id,
                    panel_id=panel_id,
                    audio_path=path,
                    duration_sec=duration,
                ))
        ctx.audio_segments = segments
        logger.info("Restored %d audio segments from %s", len(segments), ctx.audio_dir)

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

        # Select device and dtype
        if torch.cuda.is_available():
            device = "cuda:0"
            dtype = torch.bfloat16
        else:
            device = "cpu"
            dtype = torch.float32
            logger.warning("CUDA not available, using CPU for TTS (will be slow)")

        if torch.cuda.is_available():
            # TF32 — faster matmul on Ampere/Ada without precision loss for TTS
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            # cuDNN autotuner — finds fastest conv algorithm for fixed input sizes
            torch.backends.cudnn.benchmark = True
            # Explicit SDPA kernel priority: Flash SDP → Mem-efficient SDP → Math
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
            logger.info("CUDA optimizations enabled (TF32, cudnn.benchmark, SDPA)")

        # FlashAttention 2 if installed (Linux only), otherwise SDPA
        attn_impl = "eager"
        if torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                logger.info("FlashAttention 2 enabled")
            except ImportError:
                attn_impl = "sdpa"
                logger.info("Using PyTorch SDPA attention")

        try:
            model = Qwen3TTSModel.from_pretrained(
                cfg.tts_model,
                device_map=device,
                dtype=dtype,
                attn_implementation=attn_impl,
            )
        except TypeError:
            model = Qwen3TTSModel.from_pretrained(
                cfg.tts_model,
                device_map=device,
                dtype=dtype,
            )
            logger.info("Model loaded without custom attention implementation")
        except Exception as exc:
            raise TTSError(f"Failed to load TTS model '{cfg.tts_model}': {exc}") from exc

        # torch.compile — try default mode (more compatible than reduce-overhead)
        if torch.cuda.is_available() and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="default", fullgraph=False)
                logger.info("torch.compile enabled")
            except Exception as exc:
                logger.warning("torch.compile skipped: %s", exc)

        audio_dir = ctx.audio_dir
        audio_dir.mkdir(parents=True, exist_ok=True)

        import soundfile as sf

        segments: list[AudioSegment] = []
        n_segments = sum(
            len(s.panel_narrations) if s.panel_narrations else 1
            for s in scenes
        )
        done = 0

        for scene in scenes:
            # Determine what to voice: per-panel or per-scene
            items: list[tuple[str, str, str]]  # (text, out_path_stem, panel_id)
            if scene.panel_narrations:
                items = [
                    (pn.text, f"panel_{pn.panel_id}", pn.panel_id)
                    for pn in scene.panel_narrations
                ]
            else:
                items = [(scene.narration, f"scene_{scene.scene_id:03d}", "")]

            for text, stem, panel_id in items:
                progress.on_stage_progress(
                    self.name, done, n_segments,
                    f"Озвучка {done + 1}/{n_segments}...",
                )
                out_path = audio_dir / f"{stem}.wav"

                if out_path.exists():
                    duration = _wav_duration(out_path)
                    segments.append(AudioSegment(
                        scene_id=scene.scene_id,
                        panel_id=panel_id,
                        audio_path=out_path,
                        duration_sec=duration,
                    ))
                    done += 1
                    continue

                text = text.strip()
                if not text:
                    logger.warning("Empty text for %s, skipping", stem)
                    done += 1
                    continue

                try:
                    with torch.inference_mode():
                        wavs, sr = model.generate_custom_voice(
                            text=text,
                            language=cfg.tts_language,
                            speaker=cfg.tts_speaker,
                        )
                except Exception as exc:
                    raise TTSError(f"TTS failed for {stem}: {exc}") from exc

                sf.write(str(out_path), wavs[0], sr)
                duration = _wav_duration(out_path)
                segments.append(AudioSegment(
                    scene_id=scene.scene_id,
                    panel_id=panel_id,
                    audio_path=out_path,
                    duration_sec=duration,
                ))
                logger.info("Voiced %s: %.1fs (%d chars)", stem, duration, len(text))
                done += 1

        ctx.audio_segments = segments

        progress.on_stage_progress(self.name, total, total, "Готово")

        total_dur = sum(s.duration_sec for s in segments)
        logger.info(
            "Voiceover complete: %d segments, total %.1fs",
            len(segments), total_dur,
        )
