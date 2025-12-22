"""docTR text detection and recognition plugin.

Uses Mindee's docTR library with flexible model architecture selection.
Supports both TensorFlow and PyTorch backends (PyTorch recommended).
Returns detected text lines as tags with confidence scores.
Bounding boxes are included in metadata for each detection.

License: Apache 2.0 - fully open for commercial use.
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

# Available model architectures for reference
DET_ARCHITECTURES = [
    "db_resnet50", "db_mobilenet_v3_large",
    "linknet_resnet18", "linknet_resnet34", "linknet_resnet50",
    "fast_tiny", "fast_small", "fast_base"
]

RECO_ARCHITECTURES = [
    "crnn_vgg16_bn", "crnn_mobilenet_v3_small", "crnn_mobilenet_v3_large",
    "sar_resnet31", "master", "vitstr_small", "vitstr_base",
    "parseq", "viptr_tiny"
]

# Default configuration
DEFAULT_CONFIG = {
    "det_arch": "db_resnet50",
    "reco_arch": "crnn_vgg16_bn",
    "threshold": 0.5,
    "limit": 100,
    "include_boxes": True,
    "assume_straight_pages": True,
    "preserve_aspect_ratio": True,
    "sort_by": "confidence",
}


class DocTRPlugin(PluginBase):
    """docTR OCR plugin for document text detection and recognition."""

    def __init__(self, plugin_dir: Path):
        """Initialize the docTR plugin."""
        super().__init__(plugin_dir)
        self._model = None
        self._config = DEFAULT_CONFIG.copy()

    def get_info(self) -> PluginInfo:
        """Return plugin metadata."""
        return PluginInfo(
            name="doctr",
            version=PLUGIN_VERSION,
            description="docTR - Deep learning OCR with flexible model selection",
            hardware_reqs={
                "gpu": False,  # Works on CPU, but GPU recommended
                "min_ram_gb": 4,
            },
            provides_confidence=True,
            recommended_threshold=0.5,
        )

    def is_available(self) -> bool:
        """Check if python-doctr is installed."""
        try:
            from doctr.models import ocr_predictor
            return True
        except ImportError:
            return False

    def setup(self) -> bool:
        """Install python-doctr package with PyTorch backend."""
        try:
            import subprocess

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "python-doctr[torch]"],
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
            det_arch: Detection architecture (db_resnet50, fast_tiny, etc.)
            reco_arch: Recognition architecture (crnn_vgg16_bn, master, etc.)
            threshold: Minimum confidence to return
            limit: Max text lines to return
            include_boxes: Include bounding boxes in metadata
            assume_straight_pages: Optimize for non-rotated pages
            preserve_aspect_ratio: Keep aspect ratio when resizing
            sort_by: Sort order (confidence, position, alphabetical)
        """
        # Track if we need to reload model
        needs_reload = False
        reload_triggers = {"det_arch", "reco_arch", "assume_straight_pages", "preserve_aspect_ratio"}

        for key, value in kwargs.items():
            if key in self._config:
                if key in reload_triggers and self._config[key] != value:
                    needs_reload = True
                self._config[key] = value

        # Reload model if critical settings changed
        if needs_reload and self._model is not None:
            self._model = None  # Will reload on next tag() call

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
                "docTR not installed. Run 'pip install python-doctr[torch]'"
            )

        # Lazy load model
        if self._model is None:
            self._load_model()

        # Run inference
        start_time = time.perf_counter()
        tags, metadata = self._run_inference(image_path)
        inference_time = (time.perf_counter() - start_time) * 1000

        # Include model info in name
        model_name = f"doctr_{self._config['det_arch']}_{self._config['reco_arch']}"

        return TagResult(
            tags=tags,
            model=model_name,
            version=PLUGIN_VERSION,
            inference_time_ms=round(inference_time, 2),
            metadata=metadata,
        )

    def _load_model(self) -> None:
        """Load the docTR OCR model."""
        try:
            from doctr.models import ocr_predictor

            det_arch = self._config["det_arch"]
            reco_arch = self._config["reco_arch"]

            # Validate architectures
            if det_arch not in DET_ARCHITECTURES:
                raise PluginError(
                    f"Invalid detection architecture: {det_arch}\n"
                    f"Valid options: {', '.join(DET_ARCHITECTURES)}"
                )
            if reco_arch not in RECO_ARCHITECTURES:
                raise PluginError(
                    f"Invalid recognition architecture: {reco_arch}\n"
                    f"Valid options: {', '.join(RECO_ARCHITECTURES)}"
                )

            print(f"Loading docTR model (det={det_arch}, reco={reco_arch})...")

            self._model = ocr_predictor(
                det_arch=det_arch,
                reco_arch=reco_arch,
                pretrained=True,
                assume_straight_pages=self._config["assume_straight_pages"],
                preserve_aspect_ratio=self._config["preserve_aspect_ratio"],
            )

        except ImportError as e:
            raise PluginError(
                f"Missing dependencies. Install with:\n"
                f"pip install 'python-doctr[torch]'\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise PluginError(f"Failed to load docTR model: {e}")

    def _run_inference(self, image_path: Path) -> tuple[list[Tag], dict[str, Any]]:
        """Run OCR inference on an image.

        Returns:
            Tuple of (tags list, metadata dict)
        """
        try:
            from doctr.io import DocumentFile

            # Handle image formats that docTR might not support directly
            img_path = str(image_path)
            temp_path = None

            if image_path.suffix.lower() in {".webp", ".tiff", ".tif"}:
                from PIL import Image
                import tempfile

                img = Image.open(image_path).convert("RGB")
                temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                img.save(temp_file.name)
                img_path = temp_file.name
                temp_path = temp_file.name

            try:
                # Load image
                doc = DocumentFile.from_images(img_path)

                # Run OCR
                result = self._model(doc)
            finally:
                # Clean up temp file
                if temp_path:
                    import os
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass

            tags = []
            boxes_data = []

            # Extract text from hierarchical structure
            # Document -> Pages -> Blocks -> Lines -> Words
            for page in result.pages:
                page_h, page_w = page.dimensions

                for block in page.blocks:
                    for line in block.lines:
                        # Combine words into line text
                        line_text = " ".join(word.value for word in line.words)

                        # Calculate average confidence for the line
                        if line.words:
                            line_conf = sum(w.confidence for w in line.words) / len(line.words)
                        else:
                            line_conf = 0

                        # Apply threshold filter
                        if line_conf < self._config["threshold"]:
                            continue

                        if not line_text.strip():
                            continue

                        tags.append(Tag(
                            label=line_text.strip(),
                            confidence=round(float(line_conf), 4),
                        ))

                        if self._config["include_boxes"]:
                            # docTR coordinates are normalized (0-1)
                            # Convert to pixel coordinates
                            geom = line.geometry
                            bbox = [
                                [int(geom[0][0] * page_w), int(geom[0][1] * page_h)],
                                [int(geom[1][0] * page_w), int(geom[1][1] * page_h)],
                            ]
                            boxes_data.append({
                                "text": line_text.strip(),
                                "confidence": round(float(line_conf), 4),
                                "bbox": bbox,
                                "geometry_normalized": [list(geom[0]), list(geom[1])],
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
                "total_lines": len(tags),
                "det_arch": self._config["det_arch"],
                "reco_arch": self._config["reco_arch"],
            }

            if self._config["include_boxes"]:
                metadata["boxes"] = boxes_data

            # Add page info
            if result.pages:
                page = result.pages[0]
                metadata["page_dimensions"] = list(page.dimensions)
                if hasattr(page, 'orientation') and page.orientation:
                    metadata["orientation"] = {
                        "value": page.orientation.get("value", 0),
                        "confidence": page.orientation.get("confidence", 1.0),
                    }

            return tags, metadata

        except Exception as e:
            raise PluginError(f"OCR inference failed: {e}")

    def ocr_raw(self, image_path: Path) -> dict[str, Any]:
        """Run OCR and return full hierarchical results.

        This is an extended method not part of the PluginBase interface.
        Use this when you need the complete document structure.

        Args:
            image_path: Path to image file

        Returns:
            Dict with full document structure including pages, blocks, lines, words
        """
        if self._model is None:
            self._load_model()

        from doctr.io import DocumentFile

        doc = DocumentFile.from_images(str(image_path))
        result = self._model(doc)

        # Export to JSON-serializable dict
        return result.export()
