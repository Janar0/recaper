"""Shared test fixtures."""

from pathlib import Path

import pytest

from recaper.config import RecaperConfig
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.progress import SilentReporter


@pytest.fixture
def tmp_work_dir(tmp_path: Path) -> Path:
    return tmp_path / "work"


@pytest.fixture
def config(tmp_work_dir: Path) -> RecaperConfig:
    return RecaperConfig(
        openrouter_api_key="test-key",
        work_dir=tmp_work_dir,
    )


@pytest.fixture
def silent_reporter() -> SilentReporter:
    return SilentReporter()
