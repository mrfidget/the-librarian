"""
Plain-text file processor.
Reads the file in fixed-size chunks so that even very large text
files never occupy more than CHUNK_SIZE bytes of RAM at once.
"""
from pathlib import Path

from src.base import AbstractProcessor, FileType, FileContent
from src.config import get_config


class TextProcessor(AbstractProcessor):
    """Extracts text content from plain-text files."""

    def __init__(self):
        self.config = get_config()

    # ---------------------------------------------------------- interface

    def can_process(self, file_type: FileType) -> bool:
        """Return True only for TEXT files."""
        return file_type == FileType.TEXT

    def process(self, file_path: Path) -> FileContent:
        """
        Read the file and return its full text plus a short description.

        The description is a single-line preview of the first 200
        characters, suitable for search-result snippets.

        Args:
            file_path: Path to the text file

        Returns:
            FileContent with extracted_text and description populated
        """
        chunks = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                while True:
                    chunk = f.read(self.config.chunk_size)
                    if not chunk:
                        break
                    chunks.append(chunk)

            full_text = ''.join(chunks)

            # Build a short preview for the description field
            preview = full_text[:200].replace('\n', ' ').strip()
            if len(full_text) > 200:
                preview += '...'

            return FileContent(
                file_id=0,  # Orchestrator sets this before indexing
                extracted_text=full_text,
                description=f"Text document: {preview}",
                is_fully_redacted=False,
                page_count=None
            )

        except Exception as e:
            return FileContent(
                file_id=0,
                extracted_text=None,
                description=f"Failed to process text file: {e}",
                is_fully_redacted=False,
                page_count=None
            )
