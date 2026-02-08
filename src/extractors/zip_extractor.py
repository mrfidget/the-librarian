"""
Zip archive extractor.
Extracts one file at a time in fixed-size chunks to stay within
memory constraints even for very large archives.
"""
import zipfile
from pathlib import Path
from typing import Generator

from src.base import AbstractExtractor


class ZipExtractor(AbstractExtractor):
    """Extracts files from zip archives one entry at a time."""

    # Read/write chunk size when extracting individual entries
    CHUNK_SIZE = 65536  # 64 KB

    def extract(self, archive_path: Path, destination: Path) -> Generator[Path, None, None]:
        """
        Extract every file in the zip, yielding each extracted path.

        Directories inside the archive are recreated under destination.
        Each entry is written in CHUNK_SIZE pieces so that large files
        inside the archive never fully occupy RAM.

        Args:
            archive_path: Path to the .zip file
            destination: Root directory to extract into

        Yields:
            Full path of each extracted file

        Raises:
            zipfile.BadZipFile: If the archive is corrupt
        """
        destination.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(archive_path, 'r') as zf:
            for name in zf.namelist():
                # Skip directory entries
                if name.endswith('/'):
                    continue

                extracted_path = destination / name
                extracted_path.parent.mkdir(parents=True, exist_ok=True)

                # Stream extraction — never load the full entry into memory
                with zf.open(name) as src, open(extracted_path, 'wb') as dst:
                    while True:
                        chunk = src.read(self.CHUNK_SIZE)
                        if not chunk:
                            break
                        dst.write(chunk)

                yield extracted_path

    def is_archive(self, file_path: Path) -> bool:
        """
        Determine whether a file is a zip archive.

        Checks the file extension first (fast path), then falls back to
        inspecting the first two magic bytes ('PK').

        Args:
            file_path: Path to inspect

        Returns:
            True if the file appears to be a zip archive
        """
        if not file_path.exists() or not file_path.is_file():
            return False

        # Fast path: extension check
        if file_path.suffix.lower() == '.zip':
            return True

        # Magic-byte check — ZIP starts with 'PK'
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                # PK\x03\x04  — normal zip
                # PK\x05\x06  — empty zip
                return header[:2] == b'PK' and header[2:4] in (b'\x03\x04', b'\x05\x06')
        except Exception:
            return False
