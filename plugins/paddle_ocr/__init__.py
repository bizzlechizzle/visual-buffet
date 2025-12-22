"""PaddleOCR text detection and recognition plugin.

Uses PaddleOCR for high-speed OCR with 100+ language support.
Returns detected text lines as tags with confidence scores.
Bounding boxes are included in metadata for each detection.
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
# Note: PaddleOCR v3 renamed parameters:
#   use_angle_cls -> use_textline_orientation
#   det_db_thresh -> text_det_thresh
#   det_db_box_thresh -> text_det_box_thresh
#   det_db_unclip_ratio -> text_det_unclip_ratio
#   drop_score -> text_rec_score_thresh
DEFAULT_CONFIG = {
    "language": "en",
    "use_textline_orientation": True,
    "text_det_thresh": 0.3,
    "text_det_box_thresh": 0.5,
    "text_det_unclip_ratio": 1.5,
    "text_rec_score_thresh": 0.5,
    "threshold": 0.5,
    "limit": 100,
    "sort_by": "confidence",
    "include_boxes": True,
}


class PaddleOCRPlugin(PluginBase):
    """PaddleOCR text detection and recognition plugin."""

    def __init__(self, plugin_dir: Path):
        """Initialize the PaddleOCR plugin."""
        super().__init__(plugin_dir)
        self._ocr = None
        self._config = DEFAULT_CONFIG.copy()

    def get_info(self) -> PluginInfo:
        """Return plugin metadata."""
        return PluginInfo(
            name="paddle_ocr",
            version=PLUGIN_VERSION,
            description="PaddleOCR - High-speed text detection and OCR with 100+ languages",
            hardware_reqs={
                "gpu": False,  # Works on CPU, but GPU recommended
                "min_ram_gb": 4,
            },
            provides_confidence=True,
            recommended_threshold=0.5,
        )

    def is_available(self) -> bool:
        """Check if paddleocr is installed."""
        try:
            import paddleocr
            return True
        except ImportError:
            return False

    def setup(self) -> bool:
        """Install paddlepaddle and paddleocr packages."""
        try:
            import subprocess

            # Install PaddlePaddle (CPU version by default)
            # Users with GPU should install paddlepaddle-gpu manually
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "paddlepaddle", "paddleocr"],
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
            language: OCR language (en, ch, fr, etc.)
            use_textline_orientation: Enable rotated text detection
            text_det_thresh: Detection threshold
            text_det_box_thresh: Box confidence threshold
            text_det_unclip_ratio: Region expansion ratio
            text_rec_score_thresh: Drop results below this score
            threshold: Minimum confidence to return
            limit: Max text lines to return
            sort_by: Sort order (confidence, position, alphabetical)
            include_boxes: Include bounding boxes in metadata
        """
        # Track if we need to reload model
        needs_reload = False
        reload_triggers = {"language"}

        for key, value in kwargs.items():
            if key in self._config:
                if key in reload_triggers and self._config[key] != value:
                    needs_reload = True
                self._config[key] = value

        # Reload model if critical settings changed
        if needs_reload and self._ocr is not None:
            self._ocr = None  # Will reload on next tag() call

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
                "PaddleOCR not installed. Run 'pip install paddlepaddle paddleocr'"
            )

        # Lazy load model
        if self._ocr is None:
            self._load_model()

        # Run inference
        start_time = time.perf_counter()
        tags, metadata = self._run_inference(image_path)
        inference_time = (time.perf_counter() - start_time) * 1000

        return TagResult(
            tags=tags,
            model=f"paddle_ocr_{self._config['language']}",
            version=PLUGIN_VERSION,
            inference_time_ms=round(inference_time, 2),
            metadata=metadata,
        )

    def _load_model(self) -> None:
        """Load the PaddleOCR model."""
        try:
            from paddleocr import PaddleOCR

            # Check GPU availability
            try:
                import paddle
                has_gpu = paddle.device.is_compiled_with_cuda()
            except Exception:
                has_gpu = False

            device = "GPU" if has_gpu else "CPU"
            print(f"Loading PaddleOCR ({self._config['language']}) on {device}...")

            # Initialize PaddleOCR with configuration (v3 API)
            # Note: PaddleOCR v3 handles device selection automatically
            self._ocr = PaddleOCR(
                lang=self._config["language"],
                use_textline_orientation=self._config["use_textline_orientation"],
                text_det_thresh=self._config["text_det_thresh"],
                text_det_box_thresh=self._config["text_det_box_thresh"],
                text_det_unclip_ratio=self._config["text_det_unclip_ratio"],
                text_rec_score_thresh=self._config["text_rec_score_thresh"],
            )

        except ImportError as e:
            raise PluginError(
                f"Missing dependencies. Install with:\n"
                f"pip install paddlepaddle paddleocr\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise PluginError(f"Failed to load PaddleOCR model: {e}")

    def _run_inference(self, image_path: Path) -> tuple[list[Tag], dict[str, Any]]:
        """Run OCR inference on an image.

        Returns:
            Tuple of (tags list, metadata dict)
        """
        try:
            # Convert image to supported format if needed (PaddleOCR v3 doesn't support webp)
            img_str = str(image_path)
            temp_path = None

            if image_path.suffix.lower() in {".webp", ".tiff", ".tif"}:
                from PIL import Image
                import tempfile

                img = Image.open(image_path).convert("RGB")
                temp_path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                img.save(temp_path.name)
                img_str = temp_path.name

            try:
                # Run PaddleOCR v3 (uses predict method)
                result = self._ocr.predict(img_str)
            finally:
                # Clean up temp file
                if temp_path:
                    import os
                    os.unlink(temp_path.name)

            tags = []
            boxes_data = []

            # Process results - PaddleOCR v3 format:
            # result[0] is an OCRResult with rec_texts, rec_scores, rec_polys
            if result and len(result) > 0:
                r = result[0]
                texts = r.get("rec_texts") or []
                scores = r.get("rec_scores") or []
                polys = r.get("rec_polys") or []

                for text, confidence, poly in zip(texts, scores, polys):
                    # Apply threshold filter
                    if confidence < self._config["threshold"]:
                        continue

                    # Create tag for this text line
                    tags.append(Tag(
                        label=text.strip(),
                        confidence=round(float(confidence), 4),
                    ))

                    # Store box data for metadata
                    if self._config["include_boxes"]:
                        # Convert numpy array to list of points
                        bbox = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in poly]
                        boxes_data.append({
                            "text": text.strip(),
                            "confidence": round(float(confidence), 4),
                            "bbox": bbox,
                        })

            # Apply limit
            limit = self._config["limit"]
            if limit > 0:
                tags = tags[:limit]
                boxes_data = boxes_data[:limit]

            # Sort results
            sort_by = self._config["sort_by"]
            if sort_by == "confidence":
                tags.sort(key=lambda t: -(t.confidence or 0))
                boxes_data.sort(key=lambda b: -b["confidence"])
            elif sort_by == "alphabetical":
                tags.sort(key=lambda t: t.label.lower())
                boxes_data.sort(key=lambda b: b["text"].lower())
            # "position" keeps original order (reading order from PaddleOCR)

            # Build metadata
            metadata = {
                "language": self._config["language"],
                "total_lines": len(tags),
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
        if self._ocr is None:
            self._load_model()

        # Convert image to supported format if needed
        img_str = str(image_path)
        temp_path = None

        if image_path.suffix.lower() in {".webp", ".tiff", ".tif"}:
            from PIL import Image
            import tempfile

            img = Image.open(image_path).convert("RGB")
            temp_path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            img.save(temp_path.name)
            img_str = temp_path.name

        try:
            result = self._ocr.predict(img_str)
        finally:
            if temp_path:
                import os
                os.unlink(temp_path.name)

        detections = []
        if result and len(result) > 0:
            r = result[0]
            texts = r.get("rec_texts") or []
            scores = r.get("rec_scores") or []
            polys = r.get("rec_polys") or []

            for text, confidence, poly in zip(texts, scores, polys):
                bbox = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in poly]
                detections.append({
                    "text": text,
                    "confidence": round(float(confidence), 4),
                    "bbox": bbox,
                })

        return detections
