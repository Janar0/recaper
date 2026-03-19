"""Tests for VoiceoverStage."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from recaper.models import AudioSegment, ContentType, NarrativeScript, SceneBlock
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.stages.voiceover import VoiceoverStage, _wav_duration


def _make_ctx(config, tmp_path: Path) -> PipelineContext:
    ctx = PipelineContext(config=config, source_path=Path("test.cbz"))
    ctx.ensure_dirs()
    ctx.script = NarrativeScript(
        title="Test",
        content_type=ContentType.MANGA,
        scenes=[
            SceneBlock(scene_id=1, narration="Привет мир", panel_ids=["p001_001"]),
            SceneBlock(scene_id=2, narration="Второй блок", panel_ids=["p001_002"]),
        ],
        total_panels=2,
    )
    return ctx


def test_voiceover_stage_name():
    stage = VoiceoverStage()
    assert stage.name == "voiceover"
    assert stage.description == "Озвучка сцен (TTS)"


def test_is_complete_false_without_script(config):
    ctx = PipelineContext(config=config, source_path=Path("test.cbz"))
    stage = VoiceoverStage()
    assert stage.is_complete(ctx) is False


def test_is_complete_true_when_all_wavs_exist(config, tmp_path):
    ctx = _make_ctx(config, tmp_path)
    stage = VoiceoverStage()

    # Create fake wav files
    for scene in ctx.script.scenes:
        wav_path = ctx.audio_dir / f"scene_{scene.scene_id:03d}.wav"
        _write_silence_wav(wav_path)

    assert stage.is_complete(ctx) is True


def test_is_complete_false_when_wav_missing(config, tmp_path):
    ctx = _make_ctx(config, tmp_path)
    stage = VoiceoverStage()

    # Only create first wav
    wav_path = ctx.audio_dir / "scene_001.wav"
    _write_silence_wav(wav_path)

    assert stage.is_complete(ctx) is False


@pytest.mark.asyncio
async def test_skips_when_no_script(config, silent_reporter):
    ctx = PipelineContext(config=config, source_path=Path("test.cbz"))
    ctx.ensure_dirs()
    stage = VoiceoverStage()

    await stage.run(ctx, silent_reporter)
    assert ctx.audio_segments == []


@pytest.mark.asyncio
async def test_raises_without_qwen_tts(config, tmp_path, silent_reporter):
    ctx = _make_ctx(config, tmp_path)
    stage = VoiceoverStage()

    with patch.dict("sys.modules", {"qwen_tts": None, "torch": MagicMock()}):
        with pytest.raises(Exception):
            await stage.run(ctx, silent_reporter)


def _write_silence_wav(path: Path, duration_sec: float = 0.5, sample_rate: int = 24000) -> None:
    """Write a minimal silent WAV file."""
    import struct
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))


def test_wav_duration(tmp_path):
    path = tmp_path / "test.wav"
    _write_silence_wav(path, duration_sec=1.0)
    dur = _wav_duration(path)
    assert abs(dur - 1.0) < 0.01
