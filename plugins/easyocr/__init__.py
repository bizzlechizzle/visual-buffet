"""EasyOCR text detection and recognition plugin.

Uses EasyOCR's CRAFT detection + CRNN recognition pipeline.
Designed for scene text (signs, labels, photos) - NOT document OCR.
Supports 80+ languages with excellent accuracy on natural images.

This is the proper complement to PaddleOCR for cross-validation,
as both handle scene text but use different model architectures.

License: Apache 2.0
"""

import sys
import time
from pathlib import Path
from typing import Any

# Add src to path for imports during plugin loading
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from visual_buffet.exceptions import PluginError
from visual_buffet.plugins.base import PluginBase
from visual_buffet.plugins.schemas import PluginInfo, Tag, TagResult

# Version of this plugin
PLUGIN_VERSION = "1.0.0"

# Default configuration
DEFAULT_CONFIG = {
    "languages": ["en"],
    "threshold": 0.3,  # Lower default for scene text
    "limit": 100,
    "include_boxes": True,
    "paragraph": False,  # Group text into paragraphs
    "detail": 1,  # 0=simple, 1=standard
    "decoder": "greedy",  # greedy or beamsearch
    "beamWidth": 5,
    "contrast_ths": 0.1,
    "adjust_contrast": 0.5,
    "text_threshold": 0.7,
    "low_text": 0.4,
    "link_threshold": 0.4,
    "sort_by": "confidence",
}


class EasyOCRPlugin(PluginBase):
    """EasyOCR scene text detection and recognition plugin."""

    def __init__(self, plugin_dir: Path):
        """Initialize the EasyOCR plugin."""
        super().__init__(plugin_dir)
        self._reader = None
        self._config = DEFAULT_CONFIG.copy()
        self._current_languages = None

    def get_info(self) -> PluginInfo:
        """Return plugin metadata."""
        return PluginInfo(
            name="easyocr",
            version=PLUGIN_VERSION,
            description="EasyOCR - CRAFT+CRNN scene text recognition with 80+ languages",
            hardware_reqs={
                "gpu": False,  # Works on CPU, but GPU recommended
                "min_ram_gb": 4,
            },
            provides_confidence=True,
            recommended_threshold=0.3,
        )

    def is_available(self) -> bool:
        """Check if easyocr is installed."""
        try:
            import easyocr
            return True
        except ImportError:
            return False

    def setup(self) -> bool:
        """Install easyocr package."""
        try:
            import subprocess

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "easyocr"],
                check=True,
                capture_output=True,
            )
            return True
        except Exception as e:
            print(f"Setup failed: {e}")
            return False

    def configure(self, **kwargs: Any) -> None:
        """Configure plugin settings at runtime.

        Args:
            languages: List of language codes (e.g., ["en", "es"])
            threshold: Minimum confidence to return
            limit: Max text lines to return
            include_boxes: Include bounding boxes in metadata
            paragraph: Group text into paragraphs
            detail: 0=simple output, 1=standard with boxes
            decoder: "greedy" or "beamsearch"
            beamWidth: Width for beam search decoder
            contrast_ths: Contrast threshold for low contrast images
            adjust_contrast: Contrast adjustment factor
            text_threshold: Text confidence threshold
            low_text: Low text bound
            link_threshold: Link threshold for text regions
            sort_by: Sort order (confidence, position, alphabetical)
        """
        # Track if we need to reload model
        needs_reload = False

        for key, value in kwargs.items():
            if key in self._config:
                if key == "languages" and self._config[key] != value:
                    needs_reload = True
                self._config[key] = value

        # Reload model if languages changed
        if needs_reload and self._reader is not None:
            self._reader = None

    def tag(self, image_path: Path) -> TagResult:
        """Detect and recognize text in an image.

        Args:
            image_path: Path to image file

        Returns:
            TagResult with each detected text line as a Tag,
            plus bounding boxes in metadata
        """
        if not self.is_available():
            raise PluginError(
                "EasyOCR not installed. Run 'pip install easyocr'"
            )

        # Lazy load model
        if self._reader is None or self._current_languages != self._config["languages"]:
            self._load_model()

        # Run inference
        start_time = time.perf_counter()
        tags, metadata = self._run_inference(image_path)
        inference_time = (time.perf_counter() - start_time) * 1000

        # Include language info in model name
        lang_str = "_".join(self._config["languages"])

        return TagResult(
            tags=tags,
            model=f"easyocr_{lang_str}",
            version=PLUGIN_VERSION,
            inference_time_ms=round(inference_time, 2),
            metadata=metadata,
        )

    def _load_model(self) -> None:
        """Load the EasyOCR reader."""
        try:
            import easyocr

            languages = self._config["languages"]
            print(f"Loading EasyOCR ({', '.join(languages)})...")

            # Check GPU availability
            try:
                import torch
                has_gpu = torch.cuda.is_available()
            except ImportError:
                has_gpu = False

            self._reader = easyocr.Reader(
                languages,
                gpu=has_gpu,
                verbose=False,
            )
            self._current_languages = languages

        except ImportError as e:
            raise PluginError(
                f"Missing dependencies. Install with:\n"
                f"pip install easyocr\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise PluginError(f"Failed to load EasyOCR model: {e}")

    def _run_inference(self, image_path: Path) -> tuple[list[Tag], dict[str, Any]]:
        """Run OCR inference on an image.

        Returns:
            Tuple of (tags list, metadata dict)
        """
        try:
            # EasyOCR accepts file paths directly
            img_str = str(image_path)

            # Run EasyOCR with configured parameters
            results = self._reader.readtext(
                img_str,
                paragraph=self._config["paragraph"],
                detail=self._config["detail"],
                decoder=self._config["decoder"],
                beamWidth=self._config["beamWidth"],
                contrast_ths=self._config["contrast_ths"],
                adjust_contrast=self._config["adjust_contrast"],
                text_threshold=self._config["text_threshold"],
                low_text=self._config["low_text"],
                link_threshold=self._config["link_threshold"],
            )

            tags = []
            boxes_data = []

            # Process results
            # detail=1 returns: [[bbox, text, confidence], ...]
            # detail=0 returns: [text, ...]
            for result in results:
                if self._config["detail"] == 1:
                    bbox, text, confidence = result
                else:
                    text = result
                    confidence = 1.0
                    bbox = None

                # Apply threshold filter
                if confidence < self._config["threshold"]:
                    continue

                text = text.strip()
                if not text:
                    continue

                tags.append(Tag(
                    label=text,
                    confidence=round(float(confidence), 4),
                ))

                if self._config["include_boxes"] and bbox is not None:
                    # EasyOCR returns bbox as [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    boxes_data.append({
                        "text": text,
                        "confidence": round(float(confidence), 4),
                        "bbox": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in bbox],
                    })

            # Apply sorting
            sort_by = self._config["sort_by"]
            if sort_by == "confidence":
                tags.sort(key=lambda t: -(t.confidence or 0))
                boxes_data.sort(key=lambda b: -b["confidence"])
            elif sort_by == "alphabetical":
                tags.sort(key=lambda t: t.label.lower())
                boxes_data.sort(key=lambda b: b["text"].lower())
            # "position" keeps original reading order

            # Apply limit
            limit = self._config["limit"]
            if limit > 0:
                tags = tags[:limit]
                boxes_data = boxes_data[:limit]

            # Build metadata
            metadata = {
                "languages": self._config["languages"],
                "total_lines": len(tags),
                "decoder": self._config["decoder"],
            }

            if self._config["include_boxes"]:
                metadata["boxes"] = boxes_data

            return tags, metadata

        except Exception as e:
            raise PluginError(f"OCR inference failed: {e}")

    def ocr_raw(self, image_path: Path) -> list[dict[str, Any]]:
        """Run OCR and return raw results with all details.

        This is an extended method not part of the PluginBase interface.
        Use this when you need full control over the output.

        Args:
            image_path: Path to image file

        Returns:
            List of detection dicts with keys:
                - text: Recognized text
                - confidence: Recognition confidence
                - bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] corner points
        """
        if self._reader is None:
            self._load_model()

        results = self._reader.readtext(
            str(image_path),
            detail=1,
            paragraph=False,
        )

        detections = []
        for bbox, text, confidence in results:
            detections.append({
                "text": text,
                "confidence": round(float(confidence), 4),
                "bbox": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in bbox],
            })

        return detections
