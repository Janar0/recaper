"""Tests for UnpackStage."""

import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from recaper.exceptions import UnpackError
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.stages.unpack import UnpackStage, _is_image


# ---------------------------------------------------------------------------
# _is_image helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("page01.jpg", True),
    ("page01.PNG", True),
    ("page01.webp", True),
    ("Thumbs.db", False),
    (".DS_Store", False),
    ("__MACOSX/page.jpg", False),
    ("page.txt", False),
    ("page.pdf", False),
])
def test_is_image(name, expected):
    assert _is_image(name) == expected


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_png_bytes(color: tuple = (255, 0, 0), size: tuple = (10, 10)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def _make_cbz(dest: Path, filenames: list[str]) -> Path:
    cbz_path = dest / "chapter.cbz"
    with zipfile.ZipFile(cbz_path, "w") as zf:
        for name in filenames:
            zf.writestr(name, _make_png_bytes())
    return cbz_path


@pytest.fixture
def stage():
    return UnpackStage()


@pytest.fixture
def ctx(config, tmp_path):
    source = tmp_path / "source.cbz"
    source.touch()
    return PipelineContext(config=config, source_path=source)


# ---------------------------------------------------------------------------
# UnpackStage.is_complete
# ---------------------------------------------------------------------------

def test_is_complete_false_when_pages_dir_missing(stage, ctx):
    assert not stage.is_complete(ctx)


def test_is_complete_false_when_pages_dir_empty(stage, ctx):
    ctx.pages_dir.mkdir(parents=True)
    assert not stage.is_complete(ctx)


def test_is_complete_true_when_png_exists(stage, ctx):
    ctx.pages_dir.mkdir(parents=True)
    (ctx.pages_dir / "001.png").write_bytes(_make_png_bytes())
    assert stage.is_complete(ctx)


# ---------------------------------------------------------------------------
# UnpackStage.run — from directory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unpack_from_directory(stage, config, silent_reporter, tmp_path):
    img_dir = tmp_path / "pages"
    img_dir.mkdir()
    (img_dir / "p1.png").write_bytes(_make_png_bytes((255, 0, 0)))
    (img_dir / "p2.jpg").write_bytes(_make_png_bytes((0, 255, 0)))

    ctx = PipelineContext(config=config, source_path=img_dir)
    ctx.ensure_dirs()

    await stage.run(ctx, silent_reporter)

    assert len(ctx.pages) == 2
    assert all(p.suffix == ".png" for p in ctx.pages)
    assert all(p.exists() for p in ctx.pages)


@pytest.mark.asyncio
async def test_unpack_empty_directory_raises(stage, config, silent_reporter, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    ctx = PipelineContext(config=config, source_path=empty_dir)
    ctx.ensure_dirs()

    with pytest.raises(UnpackError, match="No images found"):
        await stage.run(ctx, silent_reporter)


# ---------------------------------------------------------------------------
# UnpackStage.run — from CBZ
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unpack_cbz(stage, config, silent_reporter, tmp_path):
    cbz = _make_cbz(tmp_path, ["001.jpg", "002.jpg", "003.jpg"])
    ctx = PipelineContext(config=config, source_path=cbz)
    ctx.ensure_dirs()

    await stage.run(ctx, silent_reporter)

    assert len(ctx.pages) == 3
    assert all(p.exists() for p in ctx.pages)


@pytest.mark.asyncio
async def test_unpack_cbz_skips_non_image_entries(stage, config, silent_reporter, tmp_path):
    cbz_path = tmp_path / "ch.cbz"
    with zipfile.ZipFile(cbz_path, "w") as zf:
        zf.writestr("001.jpg", _make_png_bytes())
        zf.writestr("info.txt", b"some text")
        zf.writestr("Thumbs.db", b"junk")

    ctx = PipelineContext(config=config, source_path=cbz_path)
    ctx.ensure_dirs()

    await stage.run(ctx, silent_reporter)

    assert len(ctx.pages) == 1


@pytest.mark.asyncio
async def test_unpack_bad_cbz_raises(stage, config, silent_reporter, tmp_path):
    cbz_path = tmp_path / "bad.cbz"
    cbz_path.write_bytes(b"not a zip file")

    ctx = PipelineContext(config=config, source_path=cbz_path)
    ctx.ensure_dirs()

    with pytest.raises(UnpackError, match="Bad ZIP"):
        await stage.run(ctx, silent_reporter)


# ---------------------------------------------------------------------------
# UnpackStage.run — unsupported format
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unpack_unsupported_format_raises(stage, config, silent_reporter, tmp_path):
    pdf = tmp_path / "chapter.pdf"
    pdf.touch()
    ctx = PipelineContext(config=config, source_path=pdf)
    ctx.ensure_dirs()

    with pytest.raises(UnpackError, match="Unsupported"):
        await stage.run(ctx, silent_reporter)


# ---------------------------------------------------------------------------
# UnpackStage.run — pages are PNG and sorted naturally
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unpack_output_pages_are_numbered_sequentially(stage, config, silent_reporter, tmp_path):
    cbz = _make_cbz(tmp_path, ["010.jpg", "002.jpg", "001.jpg"])
    ctx = PipelineContext(config=config, source_path=cbz)
    ctx.ensure_dirs()

    await stage.run(ctx, silent_reporter)

    names = [p.name for p in ctx.pages]
    assert names == ["001.png", "002.png", "003.png"]
