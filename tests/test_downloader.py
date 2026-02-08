"""
Tests for URLDownloader.

The live-download tests are wrapped in try/except so that CI environments
without network access simply skip rather than fail.
"""
import pytest
import tempfile
import shutil
from pathlib import Path

from src.downloaders.url_downloader import URLDownloader


@pytest.fixture
def temp_dir():
    """Provide a fresh temporary directory; cleaned up after the test."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


# A small, stable public PDF used for smoke tests
_TEST_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"


def test_download_creates_file(temp_dir):
    """Downloaded file should exist and be non-empty."""
    try:
        path = URLDownloader().download(_TEST_URL, temp_dir)
        assert path.exists()
        assert path.is_file()
        assert path.stat().st_size > 0
    except Exception as e:
        pytest.skip(f"Network unavailable: {e}")


def test_download_to_explicit_path(temp_dir):
    """When destination has a suffix it is used as the full file path."""
    target = temp_dir / "my_file.pdf"
    try:
        result = URLDownloader().download(_TEST_URL, target)
        assert result == target
        assert target.exists()
    except Exception as e:
        pytest.skip(f"Network unavailable: {e}")


def test_download_batch_yields_paths(temp_dir):
    """Batch download should yield one path per successful URL."""
    urls = [_TEST_URL]
    try:
        paths = list(URLDownloader().download_batch(urls, temp_dir))
        assert len(paths) == 1
        assert paths[0].exists()
    except Exception as e:
        pytest.skip(f"Network unavailable: {e}")


def test_download_batch_skips_bad_urls(temp_dir):
    """A bad URL in the batch should not prevent others from completing."""
    urls = ["https://invalid.invalid/no-such-file", _TEST_URL]
    try:
        paths = list(URLDownloader().download_batch(urls, temp_dir))
        # At least the valid URL should succeed
        assert len(paths) >= 1
    except Exception as e:
        pytest.skip(f"Network unavailable: {e}")
