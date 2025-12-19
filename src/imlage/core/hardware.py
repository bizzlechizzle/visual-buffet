"""Hardware detection for performance scaling.

Detects CPU, RAM, and GPU capabilities. Results are cached to
~/.imlage/hardware.json to avoid re-detection on every run.

Usage:
    profile = detect_hardware()
    if profile.gpu_type == "cuda":
        print("NVIDIA GPU available!")
"""

import json
import platform
from datetime import datetime
from pathlib import Path

import psutil

from ..exceptions import HardwareDetectionError
from ..plugins.schemas import HardwareProfile

# Cache location
CACHE_DIR = Path.home() / ".imlage"
CACHE_FILE = CACHE_DIR / "hardware.json"


def detect_hardware(force_refresh: bool = False) -> HardwareProfile:
    """Detect system hardware capabilities.

    Args:
        force_refresh: If True, ignore cache and re-detect

    Returns:
        HardwareProfile with detected capabilities

    Raises:
        HardwareDetectionError: If detection fails
    """
    # Try to load from cache first
    if not force_refresh and CACHE_FILE.exists():
        try:
            return _load_cached()
        except Exception:
            pass  # Fall through to re-detect

    # Detect hardware
    profile = _detect_all()

    # Cache results
    _save_cache(profile)

    return profile


def _load_cached() -> HardwareProfile:
    """Load hardware profile from cache file."""
    with open(CACHE_FILE) as f:
        data = json.load(f)
    return HardwareProfile.from_dict(data["hardware"])


def _save_cache(profile: HardwareProfile) -> None:
    """Save hardware profile to cache file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    data = {"detected_at": datetime.now().isoformat(), "hardware": profile.to_dict()}

    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _detect_all() -> HardwareProfile:
    """Perform full hardware detection."""
    try:
        # CPU detection
        cpu_model = platform.processor() or "Unknown"
        cpu_cores = psutil.cpu_count(logical=False) or 1

        # RAM detection
        mem = psutil.virtual_memory()
        ram_total_gb = round(mem.total / (1024**3), 1)
        ram_available_gb = round(mem.available / (1024**3), 1)

        # GPU detection
        gpu_type, gpu_name, gpu_vram_gb = _detect_gpu()

        return HardwareProfile(
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            gpu_type=gpu_type,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
        )
    except Exception as e:
        raise HardwareDetectionError(f"Failed to detect hardware: {e}")


def _detect_gpu() -> tuple[str | None, str | None, float | None]:
    """Detect GPU type and capabilities.

    Returns:
        Tuple of (gpu_type, gpu_name, vram_gb)
        gpu_type is "cuda", "mps", or None
    """
    # Try CUDA first
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            # Get VRAM in GB
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024**3), 1)
            return "cuda", gpu_name, vram_gb
    except ImportError:
        pass  # torch not installed
    except Exception:
        pass  # CUDA detection failed

    # Try MPS (Apple Silicon)
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't expose VRAM info
            return "mps", "Apple Silicon", None
    except ImportError:
        pass
    except Exception:
        pass

    # No GPU
    return None, None, None


def get_recommended_batch_size(profile: HardwareProfile) -> int:
    """Get recommended batch size based on hardware.

    Args:
        profile: Detected hardware profile

    Returns:
        Recommended batch size (1, 2, 4, or 8)
    """
    if profile.gpu_type == "cuda":
        if profile.gpu_vram_gb and profile.gpu_vram_gb >= 8:
            return 8
        return 4
    elif profile.gpu_type == "mps":
        return 4
    else:
        # CPU only
        if profile.ram_available_gb >= 16:
            return 2
        return 1
