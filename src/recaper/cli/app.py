"""CLI entry point for recaper."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from recaper import __version__

app = typer.Typer(
    name="recaper",
    help="Manga/manhwa recap video generator",
    no_args_is_help=True,
)
console = Console()


@app.command()
def process(
    source: Path = typer.Argument(..., help="Input .cbz/.cbr file or directory with images"),
    output: Path = typer.Option(Path("./work"), "--output", "-o", help="Working directory"),
    title: str = typer.Option("", "--title", "-t", help="Title for the narrative"),
    model: str = typer.Option("", "--model", "-m", help="OpenRouter model override"),
    batch_size: int = typer.Option(0, "--batch-size", help="Panels per LLM request (0 = use config)"),
    resume: bool = typer.Option(False, "--resume", help="Resume from last completed stage"),
    min_importance: int = typer.Option(0, "--min-importance", help="Min panel importance (1-10) for recap; 0 = use config default"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging output"),
) -> None:
    """Process a manga/manhwa file through the recap pipeline."""
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(output / "pipeline.log", mode="a"),
        ] if output.exists() else [],
    )

    from recaper.config import RecaperConfig
    from recaper.pipeline.context import PipelineContext
    from recaper.pipeline.progress import RichProgressReporter
    from recaper.pipeline.runner import run_pipeline_sync
    from recaper.pipeline.stages.analyze import AnalyzeStage
    from recaper.pipeline.stages.detect import DetectStage
    from recaper.pipeline.stages.extract import ExtractStage
    from recaper.pipeline.stages.render import RenderStage
    from recaper.pipeline.stages.script import ScriptStage
    from recaper.pipeline.stages.unpack import UnpackStage
    from recaper.pipeline.stages.voiceover import VoiceoverStage

    if not source.exists():
        console.print(f"[red]Файл не найден:[/red] {source}")
        raise typer.Exit(1)

    # Load config from env, then apply CLI overrides
    config = RecaperConfig(work_dir=output)
    if model:
        config.openrouter_model = model
    if batch_size > 0:
        config.llm_batch_size = batch_size
    if min_importance > 0:
        config.min_panel_importance = min_importance

    # Ensure work dir exists (for log file)
    output.mkdir(parents=True, exist_ok=True)

    # Setup file logging now that dir exists
    if verbose:
        file_handler = logging.FileHandler(output / "pipeline.log", mode="a")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(file_handler)

    ctx = PipelineContext(
        config=config,
        source_path=source,
        title=title,
    )

    stages = [
        UnpackStage(),
        DetectStage(),
        ExtractStage(),
        AnalyzeStage(),
        ScriptStage(),
        VoiceoverStage(),
        RenderStage(),
    ]

    progress = RichProgressReporter(console)

    console.print(Panel(
        f"[bold]recaper[/bold] v{__version__}\n"
        f"Вход: {source}\n"
        f"Рабочая директория: {output}\n"
        f"Модель: {config.openrouter_model}\n"
        f"Этапы: {len(stages)}",
        title="MangaRecap Pipeline",
    ))

    try:
        run_pipeline_sync(stages, ctx, progress, resume=resume)
    except Exception as exc:
        console.print(f"\n[red bold]Ошибка:[/red bold] {exc}")
        raise typer.Exit(1)

    # Print summary
    console.print()
    summary_lines = [
        f"[green bold]Готово![/green bold]",
        f"Тип: {ctx.content_type.value}",
        f"Страниц: {len(ctx.pages)}",
        f"Панелей: {len(ctx.panels)}",
        f"Сцен: {len(ctx.script.scenes) if ctx.script else 0}",
        f"Сценарий: {ctx.script_path}",
    ]
    if ctx.audio_segments:
        total_audio = sum(s.duration_sec for s in ctx.audio_segments)
        summary_lines.append(f"Аудио: {len(ctx.audio_segments)} сегментов ({total_audio:.1f}с)")
    if ctx.video:
        summary_lines.append(f"Видео: {ctx.video.output_path} ({ctx.video.duration_sec:.1f}с)")
    console.print(Panel("\n".join(summary_lines), title="Результат"))


@app.command()
def version() -> None:
    """Show version."""
    console.print(f"recaper v{__version__}")


if __name__ == "__main__":
    app()
