"""
Configuration management for the document librarian system.
Loads settings from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class Config:
    """Application configuration loaded from environment variables."""

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            env_file: Optional path to .env file
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        # Processing
        self.batch_size = int(os.getenv('BATCH_SIZE', '10'))
        self.max_workers = int(os.getenv('MAX_WORKERS', '2'))
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '65536'))

        # Storage
        self.database_path = Path(os.getenv('DATABASE_PATH', '/data/database/metadata.db'))
        self.vector_db_path = Path(os.getenv('VECTOR_DB_PATH', '/data/database/vectors.db'))
        self.staging_path = Path(os.getenv('STAGING_PATH', '/data/staging'))
        self.library_path = Path(os.getenv('LIBRARY_PATH', '/data/library'))

        # Ensure directories exist
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.staging_path.mkdir(parents=True, exist_ok=True)
        
        # Create library subdirectories for each file type
        (self.library_path / 'text').mkdir(parents=True, exist_ok=True)
        (self.library_path / 'images').mkdir(parents=True, exist_ok=True)
        (self.library_path / 'pdfs').mkdir(parents=True, exist_ok=True)

        # AI Models
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self.vision_model = os.getenv('VISION_MODEL', 'openai/clip-vit-base-patch32')
        self.ocr_enabled = os.getenv('OCR_ENABLED', 'true').lower() == 'true'
        self.ocr_engine = os.getenv('OCR_ENGINE', 'tesseract')

        # Backup
        self.backup_enabled = os.getenv('BACKUP_ENABLED', 'true').lower() == 'true'
        self.backup_path = Path(os.getenv('BACKUP_PATH', '/data/backups'))
        self.backup_schedule = os.getenv('BACKUP_SCHEDULE', '0 2 * * 0')

        if self.backup_enabled:
            self.backup_path.mkdir(parents=True, exist_ok=True)

        # Resource limits
        self.max_memory_mb = int(os.getenv('MAX_MEMORY_MB', '3500'))

        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Search
        self.search_threshold = float(os.getenv('SEARCH_THRESHOLD', '0.25'))

    def __repr__(self) -> str:
        """String representation of config."""
        return (
            f"Config(batch_size={self.batch_size}, "
            f"max_workers={self.max_workers}, "
            f"database_path={self.database_path}, "
            f"embedding_model={self.embedding_model})"
        )


# Global config instance
_config: Optional[Config] = None


def get_config(env_file: Optional[str] = None) -> Config:
    """
    Get or create global config instance (singleton).

    Args:
        env_file: Optional path to .env file

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(env_file)
    return _config


def reset_config():
    """Reset global config instance (useful for testing)."""
    global _config
    _config = None