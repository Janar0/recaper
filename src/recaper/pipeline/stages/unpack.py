"""Stage: Unpack CBZ/CBR archives into individual page images."""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

from natsort import natsorted
from PIL import Image

from recaper.exceptions import UnpackError
from recaper.pipeline.context import PipelineContext
from recaper.pipeline.progress import ProgressReporter
from recaper.pipeline.stages.base import Stage

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}
SKIP_NAMES = {"thumbs.db", ".ds_store", "__macosx"}


def _is_image(name: str) -> bool:
    p = Path(name)
    if any(part.lower() in SKIP_NAMES for part in p.parts):
        return False
    return p.suffix.lower() in IMAGE_EXTENSIONS


class UnpackStage(Stage):
    @property
    def name(self) -> str:
        return "unpack"

    @property
    def description(self) -> str:
        return "Распаковка архива"

    def is_complete(self, ctx: PipelineContext) -> bool:
        return ctx.pages_dir.exists() and any(ctx.pages_dir.glob("*.png"))

    def restore(self, ctx: PipelineContext) -> None:
        ctx.pages = sorted(ctx.pages_dir.glob("*.png"))
        logger.info("Restored %d pages from %s", len(ctx.pages), ctx.pages_dir)

    async def run(self, ctx: PipelineContext, progress: ProgressReporter) -> None:
        source = ctx.source_path

        if source.is_dir():
            image_files = self._collect_from_dir(source)
        elif source.suffix.lower() in (".cbz", ".zip"):
            image_files = self._extract_zip(source, ctx.work_dir / "_raw")
        elif source.suffix.lower() in (".cbr", ".rar"):
            image_files = self._extract_rar(source, ctx.work_dir / "_raw")
        else:
            raise UnpackError(f"Unsupported input format: {source.suffix}")

        if not image_files:
            raise UnpackError("No images found in the input.")

        image_files = natsorted(image_files, key=lambda p: p.name)
        logger.info("Found %d images", len(image_files))

        ctx.pages_dir.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(image_files):
            progress.on_stage_progress(self.name, i + 1, len(image_files), f"Страница {i + 1}/{len(image_files)}")
            out_path = ctx.pages_dir / f"{i + 1:03d}.png"
            self._convert_to_png(img_path, out_path)
            ctx.pages.append(out_path)

        # Cleanup raw extraction dir
        raw_dir = ctx.work_dir / "_raw"
        if raw_dir.exists():
            shutil.rmtree(raw_dir)

        logger.info("Unpacked %d pages to %s", len(ctx.pages), ctx.pages_dir)

    def _collect_from_dir(self, directory: Path) -> list[Path]:
        return [p for p in directory.iterdir() if p.is_file() and _is_image(p.name)]

    def _extract_zip(self, archive: Path, dest: Path) -> list[Path]:
        dest.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(dest)
        except zipfile.BadZipFile as exc:
            raise UnpackError(f"Bad ZIP/CBZ file: {exc}") from exc

        return [p for p in dest.rglob("*") if p.is_file() and _is_image(p.name)]

    def _extract_rar(self, archive: Path, dest: Path) -> list[Path]:
        try:
            import rarfile
        except ImportError:
            raise UnpackError(
                "rarfile package not installed. Install with: pip install recaper[cbr]"
            )

        dest.mkdir(parents=True, exist_ok=True)
        try:
            with rarfile.RarFile(archive, "r") as rf:
                rf.extractall(dest)
        except rarfile.Error as exc:
            raise UnpackError(f"Bad RAR/CBR file: {exc}") from exc

        return [p for p in dest.rglob("*") if p.is_file() and _is_image(p.name)]

    def _convert_to_png(self, src: Path, dst: Path) -> None:
        with Image.open(src) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(dst, "PNG")
