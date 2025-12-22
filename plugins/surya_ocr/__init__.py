"""Surya OCR text detection and recognition plugin.

Uses Surya's vision transformer architecture for document OCR.
Supports 90+ languages and optional layout analysis.
Returns detected text lines as tags with confidence scores.
Bounding boxes are included in metadata for each detection.

Note: Surya is optimized for document OCR. It will NOT work well
on photographs, signs, or natural scene text.
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
    "threshold": 0.5,
    "limit": 100,
    "include_layout": False,
    "include_boxes": True,
    "det_batch_size": None,  # Auto
    "rec_batch_size": None,  # Auto
    "sort_by": "confidence",
}


class SuryaOCRPlugin(PluginBase):
    """Surya OCR plugin for document text detection and recognition."""

    def __init__(self, plugin_dir: Path):
        """Initialize the Surya OCR plugin."""
        super().__init__(plugin_dir)
        self._det_predictor = None
        self._rec_predictor = None
        self._foundation_predictor = None
        self._layout_predictor = None
        self._config = DEFAULT_CONFIG.copy()

    def get_info(self) -> PluginInfo:
        """Return plugin metadata."""
        return PluginInfo(
            name="surya_ocr",
            version=PLUGIN_VERSION,
            description="Surya OCR - Vision transformer document OCR with 90+ languages",
            hardware_reqs={
                "gpu": False,  # Works on CPU, but GPU recommended
                "min_ram_gb": 8,
            },
            provides_confidence=True,
            recommended_threshold=0.5,
        )

    def is_available(self) -> bool:
        """Check if surya-ocr is installed."""
        try:
            from surya.detection import DetectionPredictor
            return True
        except ImportError:
            return False

    def setup(self) -> bool:
        """Install surya-ocr package."""
        try:
            import subprocess

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "surya-ocr"],
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
            threshold: Minimum confidence to return
            limit: Max text lines to return
            include_layout: Run layout analysis
            include_boxes: Include bounding boxes in metadata
            det_batch_size: Detection batch size (None = auto)
            rec_batch_size: Recognition batch size (None = auto)
            sort_by: Sort order (confidence, position, alphabetical)
        """
        # Track if we need to reload layout model
        needs_layout_reload = False

        for key, value in kwargs.items():
            if key in self._config:
                if key == "include_layout" and value and not self._config[key]:
                    needs_layout_reload = True
                self._config[key] = value

        # Load layout model if newly enabled
        if needs_layout_reload and self._det_predictor is not None:
            self._load_layout_model()

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
                "Surya OCR not installed. Run 'pip install surya-ocr'"
            )

        # Lazy load models
        if self._det_predictor is None:
            self._load_models()

        # Run inference
        start_time = time.perf_counter()
        tags, metadata = self._run_inference(image_path)
        inference_time = (time.perf_counter() - start_time) * 1000

        return TagResult(
            tags=tags,
            model="surya_ocr",
            version=PLUGIN_VERSION,
            inference_time_ms=round(inference_time, 2),
            metadata=metadata,
        )

    def _load_models(self) -> None:
        """Load the Surya OCR models."""
        try:
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
            from surya.foundation import FoundationPredictor

            print("Loading Surya OCR models...")

            self._foundation_predictor = FoundationPredictor()
            self._det_predictor = DetectionPredictor()
            self._rec_predictor = RecognitionPredictor(self._foundation_predictor)

            # Load layout model if configured
            if self._config["include_layout"]:
                self._load_layout_model()

        except ImportError as e:
            raise PluginError(
                f"Missing dependencies. Install with:\n"
                f"pip install surya-ocr\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise PluginError(f"Failed to load Surya OCR models: {e}")

    def _load_layout_model(self) -> None:
        """Load the layout analysis model."""
        try:
            from surya.layout import LayoutPredictor
            from surya.foundation import FoundationPredictor
            from surya.settings import settings

            print("Loading Surya layout model...")
            self._layout_predictor = LayoutPredictor(
                FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
            )
        except Exception as e:
            print(f"Warning: Could not load layout model: {e}")
            self._layout_predictor = None

    def _run_inference(self, image_path: Path) -> tuple[list[Tag], dict[str, Any]]:
        """Run OCR inference on an image.

        Returns:
            Tuple of (tags list, metadata dict)
        """
        from PIL import Image

        try:
            # Load and prepare image
            image = Image.open(image_path).convert("RGB")

            # Surya works best with images ≤2048px width
            original_size = (image.width, image.height)
            if image.width > 2048:
                ratio = 2048 / image.width
                new_height = int(image.height * ratio)
                image = image.resize((2048, new_height), Image.LANCZOS)

            # Run OCR
            predictions = self._rec_predictor(
                [image], det_predictor=self._det_predictor
            )

            tags = []
            boxes_data = []

            # Process results
            if predictions and len(predictions) > 0:
                page = predictions[0]

                for line in page.text_lines:
                    # Get confidence (may not always be present)
                    conf = getattr(line, 'confidence', None)
                    if conf is None:
                        conf = 0.9  # Default for older Surya versions

                    # Apply threshold filter
                    if conf < self._config["threshold"]:
                        continue

                    # Get text
                    text = getattr(line, 'text', None)
                    if text is None:
                        text = str(line)
                    text = text.strip()

                    if not text:
                        continue

                    # Create tag
                    tags.append(Tag(
                        label=text,
                        confidence=round(float(conf), 4),
                    ))

                    # Store box data for metadata
                    if self._config["include_boxes"]:
                        bbox = getattr(line, 'bbox', [0, 0, 0, 0])
                        polygon = getattr(line, 'polygon', None)

                        box_entry = {
                            "text": text,
                            "confidence": round(float(conf), 4),
                            "bbox": list(bbox) if hasattr(bbox, '__iter__') else bbox,
                        }
                        if polygon is not None:
                            box_entry["polygon"] = [list(p) for p in polygon]

                        boxes_data.append(box_entry)

            # Apply limit before sorting (for efficiency)
            # Actually, sort first then limit
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
                "total_lines": len(tags),
                "original_size": original_size,
            }

            if self._config["include_boxes"]:
                metadata["boxes"] = boxes_data

            # Run layout analysis if enabled
            if self._config["include_layout"] and self._layout_predictor:
                try:
                    layout_results = self._layout_predictor([image])
                    if layout_results and len(layout_results) > 0:
                        layout_data = []
                        for elem in layout_results[0].bboxes:
                            layout_data.append({
                                "label": elem.label,
                                "bbox": list(elem.bbox),
                                "position": elem.position,
                                "confidence": getattr(elem, 'confidence', None),
                            })
                        metadata["layout"] = layout_data
                except Exception as e:
                    metadata["layout_error"] = str(e)

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
                - bbox: [x1, y1, x2, y2] bounding box
                - polygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] corner points
        """
        if self._det_predictor is None:
            self._load_models()

        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        # Resize if needed
        if image.width > 2048:
            ratio = 2048 / image.width
            new_height = int(image.height * ratio)
            image = image.resize((2048, new_height), Image.LANCZOS)

        predictions = self._rec_predictor([image], det_predictor=self._det_predictor)

        detections = []
        if predictions and len(predictions) > 0:
            for line in predictions[0].text_lines:
                text = getattr(line, 'text', str(line))
                conf = getattr(line, 'confidence', 0.9)
                bbox = getattr(line, 'bbox', [0, 0, 0, 0])
                polygon = getattr(line, 'polygon', None)

                entry = {
                    "text": text,
                    "confidence": round(float(conf), 4),
                    "bbox": list(bbox) if hasattr(bbox, '__iter__') else bbox,
                }
                if polygon is not None:
                    entry["polygon"] = [list(p) for p in polygon]

                detections.append(entry)

        return detections
