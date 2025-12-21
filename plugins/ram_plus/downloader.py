"""Model download utilities for RAM++."""

import sys
import urllib.request
from pathlib import Path


def download_file(url: str, dest: Path, desc: str = "Downloading") -> None:
    """Download a file with progress bar.

    Args:
        url: URL to download
        dest: Destination path
        desc: Description for progress bar
    """

    def progress_hook(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            percent = min(100, count * block_size * 100 // total_size)
            mb_downloaded = (count * block_size) / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r{desc}: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
        else:
            mb_downloaded = (count * block_size) / (1024 * 1024)
            sys.stdout.write(f"\r{desc}: {mb_downloaded:.1f} MB")
        sys.stdout.flush()

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
    print()  # Newline after progress


def download_model(
    model_dir: Path,
    model_url: str,
    model_name: str,
    tag_list_url: str = None,  # No longer needed - included in ram package
) -> None:
    """Download RAM++ model.

    Args:
        model_dir: Directory to save files
        model_url: URL for model weights
        model_name: Filename for model
        tag_list_url: Deprecated - tag list is included in ram package
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / model_name

    if not model_path.exists():
        print("Downloading RAM++ model (~2.9GB)...")
        print("This may take a few minutes depending on your connection.")
        download_file(model_url, model_path, "Model")
    else:
        print(f"Model already exists: {model_path}")

    print("\nSetup complete! You can now use RAM++ for tagging.")
