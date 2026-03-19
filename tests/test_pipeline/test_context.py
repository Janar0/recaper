"""Tests for PipelineContext."""

from pathlib import Path

import pytest

from recaper.pipeline.context import PipelineContext


def test_work_dir_delegates_to_config(config):
    ctx = PipelineContext(config=config, source_path=Path("test.cbz"))
    assert ctx.work_dir == config.work_dir


def test_subdirectory_properties(config):
    ctx = PipelineContext(config=config, source_path=Path("test.cbz"))
    assert ctx.pages_dir == config.work_dir / "pages"
    assert ctx.panels_dir == config.work_dir / "panels"
    assert ctx.analysis_dir == config.work_dir / "analysis"
    assert ctx.script_path == config.work_dir / "script.json"


def test_ensure_dirs_creates_subdirectories(config):
    ctx = PipelineContext(config=config, source_path=Path("test.cbz"))
    ctx.ensure_dirs()

    assert ctx.pages_dir.exists()
    assert ctx.panels_dir.exists()
    assert ctx.analysis_dir.exists()


def test_ensure_dirs_idempotent(config):
    ctx = PipelineContext(config=config, source_path=Path("test.cbz"))
    ctx.ensure_dirs()
    ctx.ensure_dirs()  # should not raise

    assert ctx.pages_dir.exists()


def test_job_id_is_unique():
    from recaper.config import RecaperConfig
    cfg = RecaperConfig(openrouter_api_key="k")
    ctx1 = PipelineContext(config=cfg, source_path=Path("a.cbz"))
    ctx2 = PipelineContext(config=cfg, source_path=Path("a.cbz"))
    assert ctx1.job_id != ctx2.job_id
