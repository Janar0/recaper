"""Tests for shared data models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from recaper.models import (
    ContentType,
    NarrativeScript,
    PanelAnalysis,
    PanelInfo,
    ReadingOrder,
    SceneBlock,
)


def test_panel_info_roundtrip(tmp_path):
    p = PanelInfo(
        panel_id="p001_002",
        page_index=0,
        panel_index=1,
        reading_order=2,
        path=tmp_path / "panel.png",
        bbox=(10, 20, 100, 200),
    )
    assert p.panel_id == "p001_002"
    assert p.is_splash is False
    assert p.is_text_only is False


def test_panel_analysis_defaults():
    pa = PanelAnalysis(panel_id="p001_001")
    assert pa.action == ""
    assert pa.characters == []
    assert pa.dialogue == []
    assert pa.importance == 5


def test_panel_analysis_importance_bounds():
    with pytest.raises(ValidationError):
        PanelAnalysis(panel_id="x", importance=0)
    with pytest.raises(ValidationError):
        PanelAnalysis(panel_id="x", importance=11)


def test_scene_block_defaults():
    sb = SceneBlock(scene_id=1, narration="Test narration", panel_ids=["p001_001"])
    assert sb.mood == "neutral"
    assert sb.pacing == "normal"
    assert sb.transition == "crossfade"


def test_narrative_script_serialisation():
    script = NarrativeScript(
        title="Chapter 1",
        content_type=ContentType.MANHWA,
        scenes=[
            SceneBlock(scene_id=1, narration="Intro", panel_ids=["p001_001"]),
        ],
        total_panels=1,
    )
    data = script.model_dump()
    assert data["title"] == "Chapter 1"
    assert data["content_type"] == "manhwa"
    assert len(data["scenes"]) == 1


def test_content_type_values():
    assert ContentType.MANGA == "manga"
    assert ContentType.MANHWA == "manhwa"
    assert ContentType.MANHUA == "manhua"


def test_reading_order_values():
    assert ReadingOrder.RTL == "rtl"
    assert ReadingOrder.TOP_DOWN == "top_down"
    assert ReadingOrder.LTR == "ltr"
