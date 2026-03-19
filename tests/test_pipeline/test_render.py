"""Tests for RenderStage."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from recaper.models import (
    AudioSegment,
    ContentType,
    NarrativeScript,
    PanelInfo,
    SceneBlock,
)
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.stages.render import (
    RenderStage,
    _crossfade_frames,
    _fit_image,
    _ken_burns_frame,
)


def test_render_stage_name():
    stage = RenderStage()
    assert stage.name == "render"
    assert stage.description == "Рендеринг видео"


def test_fit_image_wider_than_target():
    """Wide image should be letterboxed (black bars top/bottom)."""
    img = Image.new("RGB", (800, 200), (255, 0, 0))
    result = _fit_image(img, 400, 400)
    assert result.size == (400, 400)
    # Center pixel should be red
    assert result.getpixel((200, 200))[0] > 200
    # Top-left corner should be black (letterbox)
    assert result.getpixel((0, 0)) == (0, 0, 0)


def test_fit_image_taller_than_target():
    """Tall image should be pillarboxed (black bars left/right)."""
    img = Image.new("RGB", (200, 800), (0, 255, 0))
    result = _fit_image(img, 400, 400)
    assert result.size == (400, 400)
    # Center pixel should be green
    assert result.getpixel((200, 200))[1] > 200
    # Top-left corner should be black (pillarbox)
    assert result.getpixel((0, 0)) == (0, 0, 0)


def test_fit_image_exact_ratio():
    """Image with exact target ratio should fill completely."""
    img = Image.new("RGB", (800, 450), (0, 0, 255))
    result = _fit_image(img, 1920, 1080)
    assert result.size == (1920, 1080)
    # Should be all blue, no black bars
    center = result.getpixel((960, 540))
    assert center[2] > 200


def test_ken_burns_no_zoom():
    """With zoom=1.0, frame should be unchanged."""
    arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = _ken_burns_frame(arr, 0.5, 1.0, 100, 100)
    assert result.shape == arr.shape


def test_ken_burns_zoom_in():
    """With zoom>1.0, result should still be correct size."""
    arr = np.full((200, 200, 3), 128, dtype=np.uint8)
    result = _ken_burns_frame(arr, 0.5, 1.1, 200, 200)
    assert result.shape == (200, 200, 3)


def test_crossfade_full_alpha():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = np.full((10, 10, 3), 255, dtype=np.uint8)

    result = _crossfade_frames(a, b, 0.0)
    assert np.allclose(result, a)

    result = _crossfade_frames(a, b, 1.0)
    assert np.allclose(result, b)


def test_crossfade_half():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = np.full((10, 10, 3), 200, dtype=np.uint8)
    result = _crossfade_frames(a, b, 0.5)
    assert np.allclose(result, 100, atol=2)


def test_is_complete_false_no_video(config):
    ctx = PipelineContext(config=config, source_path=Path("test.cbz"))
    stage = RenderStage()
    assert stage.is_complete(ctx) is False


@pytest.mark.asyncio
async def test_skips_when_no_audio(config, silent_reporter):
    ctx = PipelineContext(config=config, source_path=Path("test.cbz"))
    ctx.ensure_dirs()
    ctx.script = NarrativeScript(
        title="Test",
        content_type=ContentType.MANGA,
        scenes=[SceneBlock(scene_id=1, narration="x", panel_ids=["p001_001"])],
        total_panels=1,
    )
    stage = RenderStage()
    await stage.run(ctx, silent_reporter)
    assert ctx.video is None
