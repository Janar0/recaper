"""Tests for PipelineRunner."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from recaper.exceptions import StageError
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.runner import PipelineRunner
from recaper.pipeline.stages.base import Stage


def _make_stage(name: str, complete: bool = False) -> Stage:
    stage = MagicMock(spec=Stage)
    stage.name = name
    stage.description = name
    stage.is_complete.return_value = complete
    stage.run = AsyncMock()
    return stage


@pytest.mark.asyncio
async def test_runner_executes_all_stages(config, silent_reporter, tmp_path):
    source = tmp_path / "ch.cbz"
    source.touch()
    ctx = PipelineContext(config=config, source_path=source)

    s1, s2 = _make_stage("a"), _make_stage("b")
    runner = PipelineRunner([s1, s2])
    await runner.run(ctx, silent_reporter)

    s1.run.assert_awaited_once()
    s2.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_runner_resume_skips_complete_stages(config, silent_reporter, tmp_path):
    source = tmp_path / "ch.cbz"
    source.touch()
    ctx = PipelineContext(config=config, source_path=source)

    s1 = _make_stage("done", complete=True)
    s2 = _make_stage("todo", complete=False)
    runner = PipelineRunner([s1, s2], resume=True)
    await runner.run(ctx, silent_reporter)

    s1.run.assert_not_awaited()
    s2.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_runner_wraps_exception_in_stage_error(config, silent_reporter, tmp_path):
    source = tmp_path / "ch.cbz"
    source.touch()
    ctx = PipelineContext(config=config, source_path=source)

    bad = _make_stage("exploder")
    bad.run = AsyncMock(side_effect=RuntimeError("boom"))

    runner = PipelineRunner([bad])
    with pytest.raises(StageError) as exc_info:
        await runner.run(ctx, silent_reporter)

    assert "exploder" in str(exc_info.value)
    assert "boom" in str(exc_info.value)


@pytest.mark.asyncio
async def test_runner_propagates_stage_error_unchanged(config, silent_reporter, tmp_path):
    source = tmp_path / "ch.cbz"
    source.touch()
    ctx = PipelineContext(config=config, source_path=source)

    bad = _make_stage("exploder")
    original = StageError("exploder", "already wrapped")
    bad.run = AsyncMock(side_effect=original)

    runner = PipelineRunner([bad])
    with pytest.raises(StageError) as exc_info:
        await runner.run(ctx, silent_reporter)

    assert exc_info.value is original
