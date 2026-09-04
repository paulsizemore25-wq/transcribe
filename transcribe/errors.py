"""Exception types shared across the package."""


class TranscribeError(Exception):
    """Base class for errors that should be reported without a traceback."""


class AudioDecodeError(TranscribeError):
    """Raised when an input file cannot be decoded into audio samples."""


class EngineError(TranscribeError):
    """Raised when the speech-recognition backend cannot be set up or run."""
