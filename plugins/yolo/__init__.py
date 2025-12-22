"""YOLO (You Only Look Once) object detection plugin.

Uses YOLOv8 from Ultralytics for real-time object detection.
Returns detected objects as tags with confidence scores.

Unlike taggers (RAM++, SigLIP), YOLO provides:
- Bounding box locations (not exposed in tag output)
- Object counts (how many of each class)
- 80 COCO classes (vs 4500+ tagger categories)
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

# Add src to path for imports during plugin loading
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from visual_buffet.exceptions import ModelNotFoundError, PluginError
from visual_buffet.plugins.base import PluginBase
from visual_buffet.plugins.schemas import PluginInfo, Tag, TagResult

# Version of this plugin
PLUGIN_VERSION = "1.0.0"

# Model variants available
MODEL_VARIANTS = {
    "n": "yolov8n.pt",  # Nano - fastest
    "s": "yolov8s.pt",  # Small
    "m": "yolov8m.pt",  # Medium - balanced
    "l": "yolov8l.pt",  # Large
    "x": "yolov8x.pt",  # Extra-large - most accurate
}

# Default variant
DEFAULT_VARIANT = "m"


class YoloPlugin(PluginBase):
    """YOLO object detection plugin."""

    def __init__(self, plugin_dir: Path):
        """Initialize the YOLO plugin."""
        super().__init__(plugin_dir)
        self._model = None
        self._device = None
        self._variant = DEFAULT_VARIANT

    def get_info(self) -> PluginInfo:
        """Return plugin metadata."""
        return PluginInfo(
            name="yolo",
            version=PLUGIN_VERSION,
            description="YOLOv8 Object Detection - 80 COCO classes with bounding boxes",
            hardware_reqs={
                "gpu": False,  # Works on CPU, but GPU recommended
                "min_ram_gb": 2,
            },
            provides_confidence=True,
            recommended_threshold=0.4,  # Filter false positives like "clock" at 0.23
        )

    def is_available(self) -> bool:
        """Check if ultralytics is installed (models auto-download)."""
        try:
            import ultralytics
            return True
        except ImportError:
            return False

    def setup(self) -> bool:
        """Install ultralytics package."""
        try:
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "ultralytics"],
                check=True,
                capture_output=True,
            )
            return True
        except Exception as e:
            print(f"Setup failed: {e}")
            return False

    def tag(self, image_path: Path) -> TagResult:
        """Detect objects in an image using YOLO."""
        if not self.is_available():
            raise PluginError(
                "Ultralytics not installed. Run 'pip install ultralytics'"
            )

        # Lazy load model
        if self._model is None:
            self._load_model()

        # Run inference
        start_time = time.perf_counter()
        tags = self._run_inference(image_path)
        inference_time = (time.perf_counter() - start_time) * 1000

        return TagResult(
            tags=tags,
            model=f"yolov8{self._variant}",
            version=PLUGIN_VERSION,
            inference_time_ms=round(inference_time, 2),
        )

    def _load_model(self) -> None:
        """Load the YOLO model."""
        try:
            import torch
            from ultralytics import YOLO

            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

            print(f"Loading YOLOv8{self._variant} model on {self._device}...")

            # Load model (auto-downloads if not cached)
            model_name = MODEL_VARIANTS.get(self._variant, MODEL_VARIANTS[DEFAULT_VARIANT])
            self._model = YOLO(model_name)

        except ImportError as e:
            raise PluginError(
                f"Missing dependencies. Install with:\n"
                f"pip install ultralytics torch torchvision\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise PluginError(f"Failed to load model: {e}")

    def _run_inference(self, image_path: Path) -> list[Tag]:
        """Run inference on an image."""
        try:
            # Run YOLO detection
            # Use very low threshold - let the engine apply user's threshold later
            results = self._model(
                str(image_path),
                device=self._device,
                verbose=False,
                conf=0.05,  # Very low - engine applies user threshold on top
            )

            # Aggregate detections by class
            # Track max confidence and count for each class
            class_stats: dict[str, dict] = defaultdict(
                lambda: {"max_conf": 0.0, "count": 0}
            )

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    label = self._model.names[cls_id]

                    class_stats[label]["count"] += 1
                    class_stats[label]["max_conf"] = max(
                        class_stats[label]["max_conf"], conf
                    )

            # Convert to Tag objects
            tags = []
            for label, stats in class_stats.items():
                # Use max confidence for the class
                tags.append(Tag(
                    label=label,
                    confidence=round(stats["max_conf"], 4),
                ))

            # Sort by confidence (descending)
            tags.sort(key=lambda t: -(t.confidence or 0))

            return tags

        except Exception as e:
            raise PluginError(f"Inference failed: {e}")

    def detect(self, image_path: Path, conf: float = 0.5) -> list[dict]:
        """Run detection and return full results with bounding boxes.

        This is an extended method not part of the PluginBase interface.
        Use this when you need location information.

        Args:
            image_path: Path to image file
            conf: Confidence threshold (0.0-1.0)

        Returns:
            List of detection dicts with keys:
                - label: Class name
                - confidence: Detection confidence
                - bbox: [x1, y1, x2, y2] pixel coordinates
                - bbox_normalized: [x1, y1, x2, y2] normalized 0-1
        """
        if self._model is None:
            self._load_model()

        results = self._model(
            str(image_path),
            device=self._device,
            verbose=False,
            conf=conf,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            # Get image dimensions for normalization
            img_h, img_w = result.orig_shape

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf_score = float(boxes.conf[i])
                label = self._model.names[cls_id]
                xyxy = boxes.xyxy[i].tolist()

                detections.append({
                    "label": label,
                    "confidence": round(conf_score, 4),
                    "bbox": [round(x, 1) for x in xyxy],
                    "bbox_normalized": [
                        round(xyxy[0] / img_w, 4),
                        round(xyxy[1] / img_h, 4),
                        round(xyxy[2] / img_w, 4),
                        round(xyxy[3] / img_h, 4),
                    ],
                })

        return detections
