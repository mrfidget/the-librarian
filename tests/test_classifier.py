"""
Tests for MagicByteClassifier.

All tests create tiny on-disk fixtures so there is no external dependency.
"""
import pytest
import tempfile
import shutil
from pathlib import Path

from src.classifiers.magic_classifier import MagicByteClassifier
from src.base import FileType


@pytest.fixture
def temp_dir():
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def classifier():
    return MagicByteClassifier()


# ------------------------------------------------------------ text files

def test_classify_txt(temp_dir, classifier):
    f = temp_dir / "hello.txt"
    f.write_text("hello world")
    assert classifier.classify(f) == FileType.TEXT


def test_classify_md(temp_dir, classifier):
    f = temp_dir / "readme.md"
    f.write_text("# Title")
    assert classifier.classify(f) == FileType.TEXT


def test_classify_csv(temp_dir, classifier):
    f = temp_dir / "data.csv"
    f.write_text("a,b\n1,2\n")
    assert classifier.classify(f) == FileType.TEXT


def test_classify_json(temp_dir, classifier):
    f = temp_dir / "config.json"
    f.write_text('{"key": "value"}')
    assert classifier.classify(f) == FileType.TEXT


# ------------------------------------------------------------ PDF (magic bytes)

def test_classify_pdf_by_extension(temp_dir, classifier):
    """A file with .pdf extension is classified as PDF even with dummy content."""
    f = temp_dir / "doc.pdf"
    # Real PDF header so libmagic also agrees
    f.write_bytes(b"%PDF-1.4 dummy content")
    assert classifier.classify(f) == FileType.PDF


# ------------------------------------------------------------ image (extension)

def test_classify_jpg_extension(temp_dir, classifier):
    """A .jpg file with minimal JPEG magic bytes."""
    f = temp_dir / "photo.jpg"
    # JPEG starts with FF D8 FF
    f.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
    assert classifier.classify(f) == FileType.IMAGE


def test_classify_png_extension(temp_dir, classifier):
    """A .png file with minimal PNG magic bytes."""
    f = temp_dir / "icon.png"
    # PNG starts with 89 50 4E 47
    f.write_bytes(b"\x89PNG" + b"\x00" * 100)
    assert classifier.classify(f) == FileType.IMAGE


# ------------------------------------------------------------ edge cases

def test_classify_nonexistent_file(temp_dir, classifier):
    assert classifier.classify(temp_dir / "ghost.txt") == FileType.UNKNOWN


def test_classify_empty_file(temp_dir, classifier):
    """An empty file with no recognisable extension returns UNKNOWN."""
    f = temp_dir / "empty"
    f.write_bytes(b"")
    assert classifier.classify(f) == FileType.UNKNOWN


def test_classify_directory_returns_unknown(temp_dir, classifier):
    """Passing a directory path should not crash."""
    assert classifier.classify(temp_dir) == FileType.UNKNOWN
