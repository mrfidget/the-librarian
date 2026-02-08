"""
File-type classifier.
Uses the file extension as a fast first pass, then falls back to
python-magic (libmagic) MIME-type detection for anything ambiguous.
"""
import magic
from pathlib import Path

from src.base import AbstractClassifier, FileType


class MagicByteClassifier(AbstractClassifier):
    """Classifies files by extension and, when needed, by MIME type."""

    # --------------- known extension sets ---------------
    TEXT_EXTENSIONS = {'.txt', '.md', '.csv', '.json', '.xml', '.log', '.rst'}
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    PDF_EXTENSION = '.pdf'

    def __init__(self):
        """Create the libmagic instance (MIME mode)."""
        self._magic = magic.Magic(mime=True)

    def classify(self, file_path: Path) -> FileType:
        """
        Return the FileType for *file_path*.

        Resolution order:
          1. Extension — instant, no I/O beyond stat.
          2. MIME type via libmagic — reads a small header from disk.
          3. UNKNOWN if neither strategy produces a match.

        Args:
            file_path: Path to the file to classify

        Returns:
            One of FileType.TEXT, IMAGE, PDF, or UNKNOWN
        """
        if not file_path.exists() or not file_path.is_file():
            return FileType.UNKNOWN

        ext = file_path.suffix.lower()

        # --- fast path: extension ---
        if ext == self.PDF_EXTENSION:
            return FileType.PDF
        if ext in self.TEXT_EXTENSIONS:
            return FileType.TEXT
        if ext in self.IMAGE_EXTENSIONS:
            return FileType.IMAGE

        # --- fallback: MIME detection ---
        try:
            mime = self._magic.from_file(str(file_path))

            if mime == 'application/pdf':
                return FileType.PDF
            if mime.startswith('text/'):
                return FileType.TEXT
            if mime.startswith('image/'):
                return FileType.IMAGE
        except Exception:
            pass

        return FileType.UNKNOWN
