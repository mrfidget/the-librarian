"""
Filesystem backup and restore.

Each backup run creates a timestamped directory under the configured
backup root.  A manifest.txt file is written alongside the copied
data so that you can see at a glance when the backup was taken and
which source paths were included.
"""
import shutil
from pathlib import Path
from typing import List
from datetime import datetime

from src.base import AbstractBackup


class FileSystemBackup(AbstractBackup):
    """Copies database files to a local backup directory."""

    def backup(self, source_paths: List[Path], destination: Path) -> bool:
        """
        Copy every path in *source_paths* into a new timestamped
        sub-directory under *destination*.

        Args:
            source_paths: Files (or directories) to back up
            destination: Root backup directory

        Returns:
            True if all copies succeeded
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = destination / f"backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)

            for src in source_paths:
                if not src.exists():
                    print(f"Warning: {src} does not exist, skipping")
                    continue

                dst = backup_dir / src.name

                if src.is_file():
                    shutil.copy2(src, dst)
                    print(f"Backed up {src} -> {dst}")
                elif src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    print(f"Backed up directory {src} -> {dst}")

            # Write a human-readable manifest
            manifest = backup_dir / "manifest.txt"
            with open(manifest, 'w') as f:
                f.write(f"Backup timestamp : {timestamp}\n")
                f.write(f"Source paths     :\n")
                for src in source_paths:
                    f.write(f"  {src}\n")

            print(f"Backup completed: {backup_dir}")
            return True

        except Exception as e:
            print(f"Backup failed: {e}")
            return False

    def restore(self, backup_path: Path, destination: Path) -> bool:
        """
        Copy every file from a previous backup directory back into
        *destination*.

        The manifest.txt file is skipped during the copy.

        Args:
            backup_path: A timestamped backup directory created by backup()
            destination: Target directory (typically the database folder)

        Returns:
            True if restore completed without error
        """
        try:
            if not backup_path.is_dir():
                print(f"Error: {backup_path} is not a backup directory")
                return False

            destination.mkdir(parents=True, exist_ok=True)

            for item in backup_path.iterdir():
                if item.name == "manifest.txt":
                    continue

                dst = destination / item.name

                if item.is_file():
                    shutil.copy2(item, dst)
                    print(f"Restored {item.name} -> {dst}")
                elif item.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(item, dst)
                    print(f"Restored directory {item.name} -> {dst}")

            print(f"Restore completed from {backup_path}")
            return True

        except Exception as e:
            print(f"Restore failed: {e}")
            return False
