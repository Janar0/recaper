"""Stage: Compose final video using ffmpeg — blurred bg, Ken Burns, per-panel audio sync."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageFilter

from recaper.exceptions import RenderError
from recaper.models import AudioSegment, VideoMeta
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.progress import ProgressReporter
from recaper.pipeline.stages.base import Stage

logger = logging.getLogger(__name__)


_OVERSCAN = 0.07  # 7% overscan gives room to pan without zoom appearing


def _compose_frame(
    panel_img: Image.Image, width: int, height: int, panel_idx: int = 0
) -> Image.Image:
    """Create blurred background + sharp panel slightly off-center, with overscan padding.

    Uses a single blur pass at overscan size to avoid redundant computation.
    """
    bg_ratio = panel_img.width / panel_img.height
    target_ratio = width / height

    # Compute overscan dimensions
    ow = int(width * (1 + _OVERSCAN))
    oh = int(height * (1 + _OVERSCAN))

    # Background: scale to fill overscan canvas, crop center, single heavy blur
    bg_ow_ratio = ow / oh
    if bg_ratio > bg_ow_ratio:
        bh = oh
        bw = int(oh * bg_ratio)
    else:
        bw = ow
        bh = int(ow / bg_ratio)
    bg = panel_img.resize((bw, bh), Image.LANCZOS)
    bx = (bw - ow) // 2
    by = (bh - oh) // 2
    bg = bg.crop((bx, by, bx + ow, by + oh))
    bg = bg.filter(ImageFilter.GaussianBlur(radius=30))

    # Foreground: scale panel to fit within the non-overscan area
    if bg_ratio > target_ratio:
        fw = width
        fh = int(width / bg_ratio)
    else:
        fh = height
        fw = int(height * bg_ratio)
    fg = panel_img.resize((fw, fh), Image.LANCZOS)

    # Offset panel slightly from center (~15% of available gap) per direction
    gap_x = width - fw
    gap_y = height - fh
    shift_x = int(gap_x * 0.15)
    shift_y = int(gap_y * 0.15)
    direction = panel_idx % 4
    cx = gap_x // 2
    cy = gap_y // 2
    offsets = [
        (cx + shift_x, cy - shift_y),   # 0: right + up
        (cx - shift_x, cy + shift_y),   # 1: left + down
        (cx - shift_x, cy - shift_y),   # 2: left + up
        (cx + shift_x, cy + shift_y),   # 3: right + down
    ]
    x_off = max(0, min(offsets[direction][0], gap_x))
    y_off = max(0, min(offsets[direction][1], gap_y))

    # Paste foreground onto overscan canvas (offset by overscan padding)
    pad_x = (ow - width) // 2
    pad_y = (oh - height) // 2
    bg.paste(fg, (pad_x + x_off, pad_y + y_off))

    return bg


def _pan_filter(panel_idx: int, frames: int, width: int, height: int, fps: int = 30) -> str:
    """Pure pan (no zoom) with smooth ease-in-out. Source must be _OVERSCAN larger."""
    if frames < 2:
        return f"scale={width}:{height}"

    ow = int(width * (1 + _OVERSCAN))
    oh = int(height * (1 + _OVERSCAN))
    pan_x = ow - width   # max horizontal travel
    pan_y = oh - height  # max vertical travel
    z = ow / width       # constant zoom factor (= 1+OVERSCAN), makes crop = output size

    # Ease-in-out via cosine: 0→1 smoothly
    ease = f"(1-cos(3.14159265*on/{frames}))/2"

    direction = panel_idx % 4
    if direction == 0:    # pan right: x 0→pan_x, y centered
        x = f"{pan_x}*({ease})"
        y = f"{pan_y // 2}"
    elif direction == 1:  # pan down: x centered, y 0→pan_y
        x = f"{pan_x // 2}"
        y = f"{pan_y}*({ease})"
    elif direction == 2:  # pan left: x pan_x→0, y centered
        x = f"{pan_x}*(1-{ease})"
        y = f"{pan_y // 2}"
    else:                 # pan up: x centered, y pan_y→0
        x = f"{pan_x // 2}"
        y = f"{pan_y}*(1-{ease})"

    return (
        f"zoompan=z={z:.5f}:x='{x}':y='{y}'"
        f":d={frames}:s={width}x{height}:fps={fps}"
    )


def _ffmpeg(*args: str) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RenderError(f"ffmpeg error:\n{result.stderr[-2000:]}")


class RenderStage(Stage):
    @property
    def name(self) -> str:
        return "render"

    @property
    def description(self) -> str:
        return "Рендеринг видео"

    def is_complete(self, ctx: PipelineContext) -> bool:
        return ctx.video_path.exists()

    async def run(self, ctx: PipelineContext, progress: ProgressReporter) -> None:
        if not ctx.script or not ctx.audio_segments:
            logger.warning("No script or audio segments, skipping render")
            return

        if not shutil.which("ffmpeg"):
            raise RenderError("ffmpeg not found in PATH")

        cfg = ctx.config
        W, H = cfg.video_width, cfg.video_height
        fps = cfg.video_fps

        scenes = ctx.script.scenes
        # Build audio lookup: panel_id → segment (or scene_id → segment for legacy)
        panel_audio: dict[str, AudioSegment] = {}
        scene_audio: dict[int, AudioSegment] = {}
        for seg in ctx.audio_segments:
            if seg.panel_id:
                panel_audio[seg.panel_id] = seg
            else:
                scene_audio[seg.scene_id] = seg

        # Build panel image lookup, skipping defective panels
        defective_ids = {
            a.panel_id for a in ctx.analyses if a.is_defective
        } if ctx.analyses else set()
        panel_paths: dict[str, Path] = {
            p.panel_id: p.path
            for p in ctx.panels
            if p.panel_id not in defective_ids
        }

        total_clips = sum(
            len(s.panel_narrations) if s.panel_narrations else len(s.effective_panel_ids())
            for s in scenes
        )
        progress.on_stage_progress(self.name, 0, total_clips, "Подготовка...")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            all_clips: list[Path] = []
            global_idx = 0

            for scene in scenes:
                # Decide panel/audio pairs
                if scene.panel_narrations:
                    pairs = [
                        (pn.panel_id, panel_audio.get(pn.panel_id))
                        for pn in scene.panel_narrations
                    ]
                else:
                    seg = scene_audio.get(scene.scene_id)
                    pids = scene.effective_panel_ids()
                    dur_each = (seg.duration_sec / len(pids)) if (seg and pids) else 3.0
                    pairs = [(pid, AudioSegment(
                        scene_id=scene.scene_id,
                        panel_id=pid,
                        audio_path=seg.audio_path if seg else Path(),
                        duration_sec=dur_each,
                    )) for pid in pids] if seg else []

                for panel_id, audio_seg in pairs:
                    progress.on_stage_progress(
                        self.name, global_idx, total_clips,
                        f"Рендер {global_idx + 1}/{total_clips}...",
                    )

                    img_path = panel_paths.get(panel_id)
                    clip_path = tmp / f"clip_{global_idx:04d}.mp4"

                    if img_path and img_path.exists() and audio_seg:
                        clip_path = self._render_panel_clip(
                            img_path, audio_seg, global_idx,
                            cfg.ken_burns_zoom, W, H, tmp, fps,
                        )
                    elif audio_seg:
                        # Black frame fallback
                        black = tmp / f"black_{global_idx}.png"
                        Image.new("RGB", (W, H), (0, 0, 0)).save(str(black))
                        clip_path = self._render_panel_clip(
                            black, audio_seg, global_idx,
                            1.0, W, H, tmp, fps,
                        )

                    if clip_path.exists():
                        all_clips.append(clip_path)
                    global_idx += 1

            if not all_clips:
                raise RenderError("No clips generated")

            progress.on_stage_progress(self.name, total_clips, total_clips, "Сборка...")
            self._concat_clips(all_clips, ctx.video_path, tmp)

        duration = self._probe_duration(ctx.video_path)
        ctx.video = VideoMeta(
            output_path=ctx.video_path,
            duration_sec=duration,
            resolution=(W, H),
            scenes_count=len(scenes),
        )
        progress.on_stage_progress(self.name, total_clips, total_clips, "Готово")
        logger.info("Video rendered: %s (%.1fs)", ctx.video_path, duration)

    def _render_panel_clip(
        self,
        img_path: Path,
        audio_seg: AudioSegment,
        panel_idx: int,
        zoom: float,   # kept for signature compat, not used (no zoom)
        width: int,
        height: int,
        tmp: Path,
        fps: int = 30,
    ) -> Path:
        # Compose blurred background + sharp panel (slightly off-center + overscan)
        composed_path = tmp / f"composed_{panel_idx:04d}.png"
        img = Image.open(img_path).convert("RGB")
        composed = _compose_frame(img, width, height, panel_idx)
        composed.save(str(composed_path))

        duration = audio_seg.duration_sec
        frames = max(2, int(duration * fps))

        vf = _pan_filter(panel_idx, frames, width, height, fps)

        video_only = tmp / f"video_{panel_idx:04d}.mp4"
        _ffmpeg(
            "-loop", "1",
            "-i", str(composed_path),
            "-vf", vf,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",
            str(video_only),
        )

        clip_path = tmp / f"clip_{panel_idx:04d}.mp4"

        if audio_seg.audio_path.exists():
            _ffmpeg(
                "-i", str(video_only),
                "-i", str(audio_seg.audio_path),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(clip_path),
            )
        else:
            shutil.copy2(video_only, clip_path)

        return clip_path

    def _concat_clips(self, clips: list[Path], output: Path, tmp: Path) -> None:
        if len(clips) == 1:
            shutil.copy2(clips[0], output)
            return
        concat_list = tmp / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{p}'" for p in clips), encoding="utf-8"
        )
        _ffmpeg(
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(output),
        )

    def _probe_duration(self, path: Path) -> float:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True,
        )
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0
