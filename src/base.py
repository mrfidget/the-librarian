"""
Base abstract classes for the document librarian system.
Defines interfaces for all major components.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Generator
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class FileType(Enum):
    """Enumeration of supported file types."""
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    UNKNOWN = "unknown"


class ProcessingState(Enum):
    """States for document processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FileMetadata:
    """Metadata for a processed file."""
    file_path: str
    original_url: str
    file_type: FileType
    file_size: int
    checksum: str
    library_path: Optional[str] = None
    processed_date: Optional[str] = None
    id: Optional[int] = None


@dataclass
class FileContent:
    """Extracted content from a file."""
    file_id: int
    extracted_text: Optional[str] = None
    description: Optional[str] = None
    is_fully_redacted: bool = False
    page_count: Optional[int] = None


@dataclass
class SearchResult:
    """Result from a search query."""
    file_id: int
    file_path: str
    description: str
    score: float
    file_type: FileType
    library_path: Optional[str] = None


class AbstractDownloader(ABC):
    """Abstract base class for downloading files."""

    @abstractmethod
    def download(self, url: str, destination: Path) -> Path:
        """
        Download a file from URL to destination.

        Args:
            url: Source URL
            destination: Destination path

        Returns:
            Path to downloaded file
        """
        pass

    @abstractmethod
    def download_batch(self, urls: List[str], destination: Path) -> Generator[Path, None, None]:
        """
        Download multiple files, yielding paths as they complete.

        Args:
            urls: List of URLs to download
            destination: Destination directory

        Yields:
            Path to each downloaded file
        """
        pass


class AbstractExtractor(ABC):
    """Abstract base class for file extraction."""

    @abstractmethod
    def extract(self, archive_path: Path, destination: Path) -> Generator[Path, None, None]:
        """
        Extract files from archive.

        Args:
            archive_path: Path to archive file
            destination: Destination directory

        Yields:
            Path to each extracted file
        """
        pass

    @abstractmethod
    def is_archive(self, file_path: Path) -> bool:
        """
        Check if file is an archive.

        Args:
            file_path: Path to check

        Returns:
            True if file is an archive
        """
        pass


class AbstractClassifier(ABC):
    """Abstract base class for file classification."""

    @abstractmethod
    def classify(self, file_path: Path) -> FileType:
        """
        Classify file type.

        Args:
            file_path: Path to file

        Returns:
            Detected file type
        """
        pass


class AbstractProcessor(ABC):
    """Abstract base class for file processors."""

    @abstractmethod
    def can_process(self, file_type: FileType) -> bool:
        """
        Check if processor can handle this file type.

        Args:
            file_type: Type of file

        Returns:
            True if processor can handle this type
        """
        pass

    @abstractmethod
    def process(self, file_path: Path) -> FileContent:
        """
        Process file and extract content.

        Args:
            file_path: Path to file

        Returns:
            Extracted file content
        """
        pass


class AbstractIndexer(ABC):
    """Abstract base class for indexing content."""

    @abstractmethod
    def index(self, file_id: int, content: FileContent) -> bool:
        """
        Index file content.

        Args:
            file_id: File identifier
            content: Content to index

        Returns:
            True if indexing successful
        """
        pass

    @abstractmethod
    def is_indexed(self, file_id: int) -> bool:
        """
        Check if file is already indexed.

        Args:
            file_id: File identifier

        Returns:
            True if file is indexed
        """
        pass


class AbstractRetriever(ABC):
    """Abstract base class for search and retrieval."""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Search for documents matching query.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of search results
        """
        pass


class AbstractBackup(ABC):
    """Abstract base class for backup operations."""

    @abstractmethod
    def backup(self, source_paths: List[Path], destination: Path) -> bool:
        """
        Backup files to destination.

        Args:
            source_paths: Paths to backup
            destination: Backup destination

        Returns:
            True if backup successful
        """
        pass

    @abstractmethod
    def restore(self, backup_path: Path, destination: Path) -> bool:
        """
        Restore from backup.

        Args:
            backup_path: Path to backup
            destination: Restore destination

        Returns:
            True if restore successful
        """
        pass