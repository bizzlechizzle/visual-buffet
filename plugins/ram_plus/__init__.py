"""RAM++ (Recognize Anything Plus Plus) plugin.

Uses the RAM++ model from https://github.com/xinyu1205/recognize-anything
for general-purpose image tagging with ~6500 possible tags.
"""

import sys
import time
from pathlib import Path

# Add src to path for imports during plugin loading
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from imlage.exceptions import ModelNotFoundError, PluginError
from imlage.plugins.base import PluginBase
from imlage.plugins.schemas import PluginInfo, Tag, TagResult

# Version of this plugin
PLUGIN_VERSION = "1.0.0"

# Model configuration
MODEL_NAME = "ram_plus_swin_large_14m.pth"
MODEL_URL = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"
TAG_LIST_URL = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_tag_list.txt"


class RamPlusPlugin(PluginBase):
    """RAM++ image tagging plugin."""

    def __init__(self, plugin_dir: Path):
        """Initialize the RAM++ plugin."""
        super().__init__(plugin_dir)
        self._model = None
        self._transform = None
        self._device = None

    def get_info(self) -> PluginInfo:
        """Return plugin metadata."""
        return PluginInfo(
            name="ram_plus",
            version=PLUGIN_VERSION,
            description="Recognize Anything Plus Plus - General purpose image tagging with ~6500 tags",
            hardware_reqs={
                "gpu": False,  # Works on CPU, but GPU recommended
                "min_ram_gb": 4,
            },
            provides_confidence=False,  # RAM++ returns tags without confidence scores
            recommended_threshold=0.0,  # No threshold needed since no confidence
        )

    def is_available(self) -> bool:
        """Check if model files are downloaded."""
        model_path = self.get_model_path() / MODEL_NAME
        # Tag list is included in the ram package, no need to check for it
        return model_path.exists()

    def setup(self) -> bool:
        """Download model files."""
        try:
            from .downloader import download_model

            download_model(
                self.get_model_path(),
                MODEL_URL,
                MODEL_NAME,
                TAG_LIST_URL,
            )
            return True
        except Exception as e:
            print(f"Setup failed: {e}")
            return False

    def tag(self, image_path: Path) -> TagResult:
        """Tag an image using RAM++."""
        if not self.is_available():
            raise ModelNotFoundError(
                "RAM++ model not found. Run 'imlage plugins setup ram_plus'"
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
            model="ram_plus_swin_large",
            version=PLUGIN_VERSION,
            inference_time_ms=round(inference_time, 2),
        )

    def _load_model(self) -> None:
        """Load the RAM++ model."""
        try:
            import torch
            from PIL import Image
            from torchvision import transforms

            # Determine device
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")

            print(f"Loading RAM++ model on {self._device}...")

            # Try to import the RAM model
            try:
                from ram import get_transform
                from ram.models import ram_plus

                model_path = self.get_model_path() / MODEL_NAME

                self._model = ram_plus(
                    pretrained=str(model_path),
                    image_size=384,
                    vit="swin_l",
                )
                self._model.eval()
                self._model = self._model.to(self._device)
                self._transform = get_transform(image_size=384)

            except ImportError:
                # Fallback: Basic model loading without ram package
                # This is a simplified version that won't work without the ram package
                raise PluginError(
                    "The 'ram' package is required. Install with:\n"
                    "pip install git+https://github.com/xinyu1205/recognize-anything.git"
                )

        except ImportError as e:
            raise PluginError(
                f"Missing dependencies. Install with:\n"
                f"pip install torch torchvision timm\n"
                f"pip install git+https://github.com/xinyu1205/recognize-anything.git\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise PluginError(f"Failed to load model: {e}")

    def _run_inference(self, image_path: Path) -> list[Tag]:
        """Run inference on an image."""
        try:
            import torch
            from PIL import Image
            from ram import inference_ram

            # Load and transform image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self._transform(image).unsqueeze(0).to(self._device)

            # Run inference
            with torch.no_grad():
                tags_str, _ = inference_ram(image_tensor, self._model)

            # Parse results - RAM++ returns pipe-separated tags string
            # Tags are ordered by relevance (first = most relevant)
            # RAM++ does NOT provide confidence scores - only tag names
            detected_tags = [t.strip() for t in tags_str.split("|") if t.strip()]

            # Create Tag objects without confidence (None indicates no score available)
            results = []
            for tag_label in detected_tags:
                results.append(Tag(label=tag_label, confidence=None))

            return results

        except Exception as e:
            raise PluginError(f"Inference failed: {e}")
