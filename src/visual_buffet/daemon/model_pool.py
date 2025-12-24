"""LRU model pool for keeping models warm in VRAM.

Manages loaded ML models with automatic eviction when VRAM threshold
is exceeded. Uses LRU (Least Recently Used) eviction policy.
"""

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Estimated VRAM usage per model (MB)
MODEL_VRAM_ESTIMATES = {
    "ram_plus": 1200,
    "siglip": 800,
    "florence_2": 1800,
    "yolo": 300,
    "blip2": 3500,
    "paddle_ocr": 400,
    "easyocr": 600,
}

# Default estimate for unknown models
DEFAULT_VRAM_MB = 1000


@dataclass
class LoadedModel:
    """Metadata for a loaded model."""

    plugin: Any
    loaded_at: datetime
    last_used: datetime
    vram_mb: int


class ModelPool:
    """LRU pool for keeping ML models warm in VRAM.

    Models are loaded on first use and kept in VRAM until the pool
    reaches its capacity threshold (default 80% of total VRAM).
    When capacity is exceeded, the least recently used model is evicted.

    Example:
        >>> pool = ModelPool(max_vram_percent=0.80)
        >>> plugin = pool.get("ram_plus", loader=lambda: load_ram_plus())
        >>> # Model is now warm, subsequent calls are fast
        >>> plugin = pool.get("ram_plus")
    """

    def __init__(self, max_vram_percent: float = 0.80):
        """Initialize model pool.

        Args:
            max_vram_percent: Maximum VRAM usage (0.0-1.0)
        """
        self.models: OrderedDict[str, LoadedModel] = OrderedDict()
        self.max_vram_percent = max_vram_percent
        self._total_vram = self._get_total_vram()
        self._max_vram = int(self._total_vram * max_vram_percent)

        logger.info(
            f"ModelPool initialized: {self._total_vram}MB total, "
            f"{self._max_vram}MB limit ({max_vram_percent:.0%})"
        )

    def _get_total_vram(self) -> int:
        """Get total VRAM in MB."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        except ImportError:
            pass
        return 0

    def _current_vram(self) -> int:
        """Get estimated current VRAM usage in MB."""
        return sum(m.vram_mb for m in self.models.values())

    def _estimate_model_size(self, name: str) -> int:
        """Estimate VRAM for a model."""
        return MODEL_VRAM_ESTIMATES.get(name, DEFAULT_VRAM_MB)

    def get(self, name: str) -> Any | None:
        """Get a loaded model by name.

        Updates last_used time and moves to end of LRU queue.

        Args:
            name: Model/plugin name

        Returns:
            Plugin instance if loaded, None otherwise
        """
        if name in self.models:
            model = self.models[name]
            model.last_used = datetime.now()
            # Move to end (most recently used)
            self.models.move_to_end(name)
            return model.plugin
        return None

    def add(self, name: str, plugin: Any, vram_mb: int | None = None) -> None:
        """Add a loaded model to the pool.

        Evicts LRU models if needed to make room.

        Args:
            name: Model/plugin name
            plugin: Plugin instance
            vram_mb: VRAM usage in MB (estimated if not provided)

        Raises:
            MemoryError: If model cannot fit even after evicting all others
        """
        if vram_mb is None:
            vram_mb = self._estimate_model_size(name)

        # If no GPU available (max_vram == 0), skip VRAM management
        if self._max_vram == 0:
            logger.warning(
                f"No GPU detected. Adding {name} to pool without VRAM limits."
            )
            self.models[name] = LoadedModel(
                plugin=plugin,
                loaded_at=datetime.now(),
                last_used=datetime.now(),
                vram_mb=vram_mb,
            )
            return

        # Evict if needed
        while self._current_vram() + vram_mb > self._max_vram:
            if not self.models:
                raise MemoryError(
                    f"Cannot fit model {name} ({vram_mb}MB) in VRAM limit ({self._max_vram}MB)"
                )
            # Evict LRU (first item)
            self._evict_lru()

        self.models[name] = LoadedModel(
            plugin=plugin,
            loaded_at=datetime.now(),
            last_used=datetime.now(),
            vram_mb=vram_mb,
        )

        logger.info(
            f"Added model {name} to pool ({vram_mb}MB). "
            f"Pool: {self._current_vram()}/{self._max_vram}MB"
        )

    def _evict_lru(self) -> str | None:
        """Evict least recently used model.

        Returns:
            Name of evicted model, or None if pool is empty
        """
        if not self.models:
            return None

        # Get LRU (first item)
        evict_name, evict_model = self.models.popitem(last=False)

        logger.info(f"Evicting model {evict_name} ({evict_model.vram_mb}MB) to free VRAM")

        # Delete plugin to free VRAM
        del evict_model.plugin

        # Clear CUDA cache
        self._clear_cuda_cache()

        return evict_name

    def _clear_cuda_cache(self) -> None:
        """Clear CUDA memory cache."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def unload(self, name: str) -> bool:
        """Unload a specific model.

        Args:
            name: Model name to unload

        Returns:
            True if model was unloaded, False if not found
        """
        if name not in self.models:
            return False

        model = self.models.pop(name)
        del model.plugin
        self._clear_cuda_cache()

        logger.info(f"Unloaded model {name}")
        return True

    async def unload_all(self) -> int:
        """Unload all models.

        Returns:
            Number of models unloaded
        """
        count = len(self.models)

        for name in list(self.models.keys()):
            model = self.models.pop(name)
            del model.plugin

        self._clear_cuda_cache()

        logger.info(f"Unloaded {count} models")
        return count

    def get_status(self) -> dict[str, Any]:
        """Get pool status for health checks.

        Returns:
            Status dict with loaded models and VRAM info
        """
        return {
            "models_loaded": [
                {
                    "name": name,
                    "vram_mb": m.vram_mb,
                    "loaded_at": m.loaded_at.isoformat(),
                    "last_used": m.last_used.isoformat(),
                }
                for name, m in self.models.items()
            ],
            "vram_used_mb": self._current_vram(),
            "vram_max_mb": self._max_vram,
            "vram_total_mb": self._total_vram,
            "vram_percent": (
                self._current_vram() / self._max_vram if self._max_vram > 0 else 0
            ),
        }

    def __len__(self) -> int:
        """Number of loaded models."""
        return len(self.models)

    def __contains__(self, name: str) -> bool:
        """Check if model is loaded."""
        return name in self.models
