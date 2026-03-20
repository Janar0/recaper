# recaper

**AI-powered manga/manhwa recap video generator.**

Automatically transforms manga chapters into narrated recap videos: extracts panels, analyzes content with vision LLMs, generates a narrative script, synthesizes voiceover, and composites the final video.

> **[Полный гайд на русском / Full guide in Russian](README.ru.md)**

## Features

- **Smart panel extraction** — YOLO-based detection with LLM vision fallback
- **Content auto-detection** — manga (B&W, RTL), manhwa (color, vertical), manhua (color, LTR)
- **AI-powered analysis** — panel-by-panel analysis via OpenRouter (Claude, Gemini, etc.)
- **Narrative script generation** — cohesive recap script with mood and pacing
- **TTS voiceover** — natural speech synthesis via Qwen3-TTS
- **Video composition** — Ken Burns effects, transitions, blurred backgrounds
- **Resumable pipeline** — 7-stage pipeline, resume from any point on failure
- **Web interface** — FastAPI-based UI for remote control and monitoring
- **Multi-format input** — CBZ, CBR, or plain image directories

## Quick Start

```bash
# Install
pip install -e .

# Set up API key
cp .env.example .env
# Edit .env and add your OpenRouter API key

# Process a manga
recaper process ./manga.cbz --output ./work --title "My Manga Recap" --verbose
```

## Requirements

- Python 3.11+
- ffmpeg (must be in PATH)
- [OpenRouter API key](https://openrouter.ai/keys)
- GPU with CUDA (optional, for local TTS)

## Installation

```bash
# Clone the repository
git clone https://github.com/Janaro/recaper.git
cd recaper

# Create virtual environment
python -m venv .venv

# Activate it
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install with desired extras
pip install -e ".[web,tts,cbr]"
```

### Optional extras

| Extra | Description |
|-------|-------------|
| `web` | FastAPI web interface |
| `tts` | Qwen3-TTS voice synthesis (requires GPU) |
| `cbr` | RAR archive support |
| `dev` | Testing tools (pytest) |

## Usage

### CLI

```bash
# Process manga/manhwa into a recap video
recaper process ./manga.cbz -o ./work -t "Title" --verbose

# Resume interrupted processing
recaper process ./manga.cbz -o ./work --resume

# Start web interface
recaper web --host 0.0.0.0 --port 8000

# Show version
recaper version
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--output, -o` | Working directory (default: `./work`) |
| `--title, -t` | Title for the narrative |
| `--model, -m` | Override OpenRouter model |
| `--batch-size` | Panels per LLM request (0 = config default) |
| `--resume` | Resume from last completed stage |
| `--min-importance` | Min panel importance 1-10 (default: 4) |
| `--verbose, -v` | Enable debug logging |

## Configuration

All settings via environment variables (prefix `RECAPER_`) or `.env` file. See [.env.example](.env.example) for the template.

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RECAPER_OPENROUTER_API_KEY` | — | **Required.** OpenRouter API key |
| `RECAPER_OPENROUTER_MODEL` | `anthropic/claude-sonnet-4-20250514` | Main LLM model |
| `RECAPER_OCR_MODEL` | `google/gemini-2.0-flash-001` | Vision model for OCR |
| `RECAPER_LANGUAGE` | `ru` | Narration language |
| `RECAPER_LLM_BATCH_SIZE` | `4` | Panels per LLM request |
| `RECAPER_VIDEO_WIDTH` | `1920` | Output video width |
| `RECAPER_VIDEO_HEIGHT` | `1080` | Output video height |

## Pipeline Stages

1. **Unpack** — extract CBZ/CBR or collect images from directory
2. **Detect** — auto-detect content type (manga/manhwa/manhua)
3. **Extract** — panel detection (YOLO + LLM fallback)
4. **Analyze** — LLM vision analysis of panels (action, characters, mood, importance)
5. **Script** — generate narrative recap script
6. **Voiceover** — TTS audio synthesis per panel
7. **Render** — compose final MP4 video with effects and transitions

## License

[GPL-3.0-or-later](LICENSE)
