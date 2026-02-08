"""
URL downloader implementation.
Downloads files from URLs with streaming to minimize memory usage.
"""
import requests
from pathlib import Path
from typing import List, Generator
from tqdm import tqdm

from src.base import AbstractDownloader
from src.config import get_config


class URLDownloader(AbstractDownloader):
    """Downloads files from URLs with streaming support."""

    def __init__(self):
        """Initialize downloader with config chunk size."""
        self.config = get_config()
        self.chunk_size = self.config.chunk_size

    def download(self, url: str, destination: Path) -> Path:
        """
        Stream-download a single file from url into destination.

        If destination is an existing directory (or does not have a suffix)
        the filename is inferred from the URL.  Otherwise destination is
        treated as the full target path.

        Args:
            url: Source URL
            destination: Directory or full file path

        Returns:
            Path to the downloaded file

        Raises:
            requests.exceptions.RequestException: on HTTP errors
        """
        # Resolve target path
        if destination.is_dir() or not destination.suffix:
            filename = url.split('/')[-1].split('?')[0] or 'download'
            destination = destination / filename

        destination.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(destination, 'wb') as f:
            if total_size:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                # Unknown total size — stream without progress bar
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)

        return destination

    def download_batch(self, urls: List[str], destination: Path) -> Generator[Path, None, None]:
        """
        Download multiple files sequentially, yielding each path on completion.

        Failures are logged to stdout and skipped so that one bad URL does
        not abort the entire batch.

        Args:
            urls: List of source URLs
            destination: Target directory

        Yields:
            Path to each successfully downloaded file
        """
        destination.mkdir(parents=True, exist_ok=True)

        for url in urls:
            try:
                downloaded_path = self.download(url, destination)
                yield downloaded_path
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                continue
