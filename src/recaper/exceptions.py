"""Custom exceptions for the recaper pipeline."""


class RecaperError(Exception):
    """Base exception for all recaper errors."""


class UnpackError(RecaperError):
    """Failed to unpack archive."""


class PanelExtractionError(RecaperError):
    """Failed to extract panels from a page."""


class LLMError(RecaperError):
    """LLM API call failed."""


class TTSError(RecaperError):
    """TTS synthesis failed."""


class RenderError(RecaperError):
    """Video rendering failed."""


class StageError(RecaperError):
    """A pipeline stage failed."""

    def __init__(self, stage_name: str, message: str) -> None:
        self.stage_name = stage_name
        super().__init__(f"Stage '{stage_name}' failed: {message}")
