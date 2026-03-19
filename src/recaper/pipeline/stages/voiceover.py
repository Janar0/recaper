"""Stage: Generate TTS audio for each scene using Qwen3-TTS."""

from __future__ import annotations

import logging
import re
import wave
from pathlib import Path

import numpy as np

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


def _normalize_text(text: str) -> str:
    """Normalize text for better TTS prosody and consistency."""
    # Collapse multiple whitespace/newlines into single space
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove markdown artifacts
    text = re.sub(r'[*_~`]', '', text)
    # Normalize ellipsis variants (2+ dots → proper ellipsis)
    text = re.sub(r'\.{2,}', '...', text)
    # Normalize dashes to natural pauses
    text = text.replace('—', ', ').replace('–', ', ')
    # Ensure sentence-ending punctuation for proper prosody
    if text and text[-1] not in '.!?…':
        text += '.'
    return text


def _normalize_audio_levels(segments: list[AudioSegment], target_peak: float = 0.9) -> None:
    """Normalize all audio segments to consistent peak volume."""
    import soundfile as sf

    for seg in segments:
        if not seg.audio_path.exists():
            continue
        data, sr = sf.read(str(seg.audio_path))
        peak = float(np.abs(data).max())
        if peak > 0 and abs(peak - target_peak) > 0.05:
            data = data * (target_peak / peak)
            data = np.clip(data, -1.0, 1.0)
            sf.write(str(seg.audio_path), data, sr)


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

        # Default sample rate for silence padding (Qwen3-TTS default)
        default_sr = 24000

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

                text = _normalize_text(text.strip())
                if not text or text == '.':
                    # Generate silence padding instead of skipping
                    logger.warning("Empty text for %s, generating silence padding", stem)
                    silence_duration = cfg.panel_padding_sec
                    silence = np.zeros(int(silence_duration * default_sr), dtype=np.float32)
                    sf.write(str(out_path), silence, default_sr)
                    segments.append(AudioSegment(
                        scene_id=scene.scene_id,
                        panel_id=panel_id,
                        audio_path=out_path,
                        duration_sec=silence_duration,
                    ))
                    done += 1
                    continue

                # TTS generation with retry
                max_retries = 2
                for attempt in range(max_retries + 1):
                    try:
                        with torch.inference_mode():
                            wavs, sr = model.generate_custom_voice(
                                text=text,
                                language=cfg.tts_language,
                                speaker=cfg.tts_speaker,
                            )
                        break
                    except Exception as exc:
                        if attempt < max_retries:
                            logger.warning("TTS attempt %d failed for %s: %s, retrying", attempt + 1, stem, exc)
                            continue
                        raise TTSError(f"TTS failed for {stem} after {max_retries + 1} attempts: {exc}") from exc

                audio_data = wavs[0]
                duration = len(audio_data) / sr
                sf.write(str(out_path), audio_data, sr)
                default_sr = sr  # remember actual sample rate for silence padding

                segments.append(AudioSegment(
                    scene_id=scene.scene_id,
                    panel_id=panel_id,
                    audio_path=out_path,
                    duration_sec=duration,
                ))
                logger.info("Voiced %s: %.1fs (%d chars)", stem, duration, len(text))
                done += 1

        # Normalize audio levels across all segments for consistent volume
        _normalize_audio_levels(segments)

        ctx.audio_segments = segments

        progress.on_stage_progress(self.name, total, total, "Готово")

        total_dur = sum(s.duration_sec for s in segments)
        logger.info(
            "Voiceover complete: %d segments, total %.1fs",
            len(segments), total_dur,
        )
