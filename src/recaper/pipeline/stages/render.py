"""Stage: Compose final video from panels and voiceover audio."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from recaper.exceptions import RenderError
from recaper.models import AudioSegment, SceneBlock, VideoMeta
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.progress import ProgressReporter
from recaper.pipeline.stages.base import Stage

logger = logging.getLogger(__name__)


def _fit_image(img: Image.Image, width: int, height: int) -> Image.Image:
    """Resize and letterbox/pillarbox an image to exactly (width, height)."""
    img_ratio = img.width / img.height
    target_ratio = width / height

    if img_ratio > target_ratio:
        # Wider than target → fit by width
        new_w = width
        new_h = int(width / img_ratio)
    else:
        # Taller than target → fit by height
        new_h = height
        new_w = int(height * img_ratio)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2
    canvas.paste(img_resized, (x_offset, y_offset))
    return canvas


def _ken_burns_frame(
    img_array: np.ndarray,
    t: float,
    zoom_factor: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Apply Ken Burns (slow zoom-in) effect at time fraction t (0..1)."""
    if zoom_factor <= 1.0:
        return img_array

    # Interpolate zoom from 1.0 to zoom_factor
    current_zoom = 1.0 + (zoom_factor - 1.0) * t
    h, w = img_array.shape[:2]

    crop_w = int(w / current_zoom)
    crop_h = int(h / current_zoom)

    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2

    cropped = img_array[y0 : y0 + crop_h, x0 : x0 + crop_w]

    # Resize back to target
    from PIL import Image as _Img

    pil = _Img.fromarray(cropped)
    pil = pil.resize((width, height), _Img.LANCZOS)
    return np.array(pil)


def _crossfade_frames(
    frame_a: np.ndarray, frame_b: np.ndarray, alpha: float
) -> np.ndarray:
    """Blend two frames: alpha=0 → frame_a, alpha=1 → frame_b."""
    return (frame_a.astype(np.float32) * (1 - alpha) + frame_b.astype(np.float32) * alpha).astype(np.uint8)


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

        cfg = ctx.config
        W, H = cfg.video_width, cfg.video_height
        fps = cfg.video_fps

        scenes = ctx.script.scenes
        audio_map: dict[int, AudioSegment] = {
            seg.scene_id: seg for seg in ctx.audio_segments
        }

        # Build panel lookup: panel_id → image path
        panel_paths: dict[str, Path] = {p.panel_id: p.path for p in ctx.panels}

        total_scenes = len(scenes)
        progress.on_stage_progress(self.name, 0, total_scenes, "Подготовка кадров...")

        try:
            import moviepy as mpy
        except ImportError as exc:
            raise RenderError("moviepy not installed. Run: pip install moviepy") from exc

        scene_clips: list = []

        for i, scene in enumerate(scenes):
            progress.on_stage_progress(
                self.name, i, total_scenes,
                f"Рендер сцены {i + 1}/{total_scenes}...",
            )

            audio_seg = audio_map.get(scene.scene_id)
            if not audio_seg:
                logger.warning("No audio for scene %d, skipping", scene.scene_id)
                continue

            scene_duration = audio_seg.duration_sec + cfg.panel_padding_sec

            # Load panel images for this scene
            images = self._load_scene_panels(scene, panel_paths, W, H)
            if not images:
                logger.warning("No panels for scene %d, using black", scene.scene_id)
                images = [np.zeros((H, W, 3), dtype=np.uint8)]

            # Create video clip for this scene with Ken Burns
            scene_clip = self._make_scene_clip(
                images, scene_duration, fps, W, H,
                cfg.ken_burns_zoom, cfg.transition_duration,
            )

            # Attach audio
            audio_clip = mpy.AudioFileClip(str(audio_seg.audio_path))
            scene_clip = scene_clip.with_audio(audio_clip)

            scene_clips.append(scene_clip)

        if not scene_clips:
            raise RenderError("No scene clips generated")

        progress.on_stage_progress(
            self.name, total_scenes, total_scenes, "Сборка видео..."
        )

        # Concatenate all scenes with crossfade transitions
        td = cfg.transition_duration
        if len(scene_clips) > 1 and td > 0:
            final = mpy.concatenate_videoclips(
                scene_clips, method="compose", padding=-td
            )
        else:
            final = mpy.concatenate_videoclips(scene_clips, method="compose")

        # Write output
        output_path = ctx.video_path
        try:
            final.write_videofile(
                str(output_path),
                fps=fps,
                codec="libx264",
                audio_codec="aac",
                preset="medium",
                threads=4,
                logger=None,
            )
        except Exception as exc:
            raise RenderError(f"Failed to write video: {exc}") from exc
        finally:
            final.close()
            for clip in scene_clips:
                clip.close()

        ctx.video = VideoMeta(
            output_path=output_path,
            duration_sec=final.duration,
            resolution=(W, H),
            scenes_count=len(scene_clips),
        )

        progress.on_stage_progress(self.name, total_scenes, total_scenes, "Готово")
        logger.info("Video rendered: %s (%.1fs)", output_path, final.duration)

    def _load_scene_panels(
        self,
        scene: SceneBlock,
        panel_paths: dict[str, Path],
        width: int,
        height: int,
    ) -> list[np.ndarray]:
        """Load and prepare panel images for a scene."""
        images = []
        for pid in scene.panel_ids:
            path = panel_paths.get(pid)
            if not path or not path.exists():
                logger.warning("Panel %s not found, skipping", pid)
                continue
            try:
                img = Image.open(path).convert("RGB")
                img = _fit_image(img, width, height)
                images.append(np.array(img))
            except Exception as exc:
                logger.warning("Failed to load panel %s: %s", pid, exc)
        return images

    def _make_scene_clip(
        self,
        images: list[np.ndarray],
        duration: float,
        fps: int,
        width: int,
        height: int,
        zoom: float,
        transition_dur: float,
    ):
        """Create a moviepy clip for one scene with Ken Burns and panel crossfades."""
        import moviepy as mpy

        n_panels = len(images)
        panel_dur = duration / n_panels if n_panels > 0 else duration

        sub_clips = []
        for img_array in images:
            def make_frame(get_frame, t, _img=img_array, _dur=panel_dur):
                frac = t / _dur if _dur > 0 else 0
                return _ken_burns_frame(_img, frac, zoom, width, height)

            clip = mpy.VideoClip(
                lambda t, _img=img_array, _dur=panel_dur: _ken_burns_frame(
                    _img, t / _dur if _dur > 0 else 0, zoom, width, height
                ),
                duration=panel_dur,
            ).with_fps(fps)
            sub_clips.append(clip)

        if len(sub_clips) == 1:
            return sub_clips[0]

        # Crossfade between panels within a scene
        inner_td = min(transition_dur * 0.5, panel_dur * 0.3)
        if inner_td > 0 and len(sub_clips) > 1:
            return mpy.concatenate_videoclips(
                sub_clips, method="compose", padding=-inner_td
            )
        return mpy.concatenate_videoclips(sub_clips, method="compose")
