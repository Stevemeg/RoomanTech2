"""Custom exceptions.

Using specific exception types (instead of letting raw ValueError / IOError
bubble up) lets callers (CLI batch loop, API handlers) distinguish "this one
resume is bad" from "the whole process is broken" and respond accordingly —
per the requirement that one malformed resume must never crash a batch run.
"""


class ResumeRankerError(Exception):
    """Base class for all application-specific errors."""


class UnsupportedFileTypeError(ResumeRankerError):
    """Raised when a resume's extension isn't in config.parser.supported_formats."""


class MimeTypeMismatchError(ResumeRankerError):
    """Raised when a file's actual content (magic bytes) doesn't match its
    claimed extension — e.g. a `.pdf` that isn't actually a PDF. Caught
    before parsing even starts, since we never trust a client-supplied
    extension or Content-Type header on its own."""


class FileTooLargeError(ResumeRankerError):
    """Raised when a resume exceeds config.parser.max_file_size_mb."""


class EmptyDocumentError(ResumeRankerError):
    """Raised when a resume contains no extractable text (e.g. blank or image-only PDF)."""


class EncryptedDocumentError(ResumeRankerError):
    """Raised when a PDF is password-protected and cannot be opened."""


class CorruptedDocumentError(ResumeRankerError):
    """Raised when a document exists but cannot be parsed (corrupt bytes, bad structure)."""


class ResumeParsingError(ResumeRankerError):
    """Generic catch-all wrapper for unexpected parsing failures on a single file."""


class EmbeddingModelUnavailableError(ResumeRankerError):
    """Raised when sentence-transformers / the configured model can't be loaded.

    Not a fatal error for the pipeline as a whole — callers (hybrid_scorer)
    catch this and fall back to TF-IDF-only similarity.
    """
