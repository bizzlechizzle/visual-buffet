"""SigLIP-based scene classification.

Uses SigLIP zero-shot classification for scene categorization
instead of tag-based rules for better accuracy.

Example:
    >>> classifier = SceneClassifier()
    >>> result = classifier.classify("photo.jpg", ["indoor", "outdoor"])
    >>> print(result)  # {"label": "outdoor", "confidence": 0.85}
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# Pre-defined scene categories with prompt templates
SCENE_CATEGORIES = {
    "indoor_outdoor": {
        "indoor": [
            "an indoor photo",
            "a photo taken inside",
            "an interior photo",
            "a photo of a room",
        ],
        "outdoor": [
            "an outdoor photo",
            "a photo taken outside",
            "an exterior photo",
            "a photo of nature",
        ],
    },
    "time_of_day": {
        "daytime": [
            "a photo taken during the day",
            "a daytime photo",
            "a photo with daylight",
        ],
        "nighttime": [
            "a photo taken at night",
            "a nighttime photo",
            "a photo with artificial lighting",
        ],
        "golden_hour": [
            "a photo taken at sunset",
            "a photo taken at sunrise",
            "a golden hour photo",
        ],
    },
    "photo_type": {
        "portrait": [
            "a portrait photo",
            "a photo of a person",
            "a headshot",
        ],
        "landscape": [
            "a landscape photo",
            "a scenic photo",
            "a wide angle photo of nature",
        ],
        "macro": [
            "a macro photo",
            "a close-up photo",
            "a detailed photo",
        ],
        "street": [
            "a street photo",
            "an urban photo",
            "a city photo",
        ],
    },
}


@dataclass
class ClassificationResult:
    """Result of scene classification."""

    label: str
    confidence: float
    all_scores: dict[str, float]
    method: str = "siglip_zeroshot"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
            "method": self.method,
        }


class SceneClassifier:
    """SigLIP-based scene classifier.

    Uses zero-shot classification with prompt ensembling for
    better accuracy than tag-based rules.

    Example:
        >>> classifier = SceneClassifier()
        >>> result = classifier.classify_indoor_outdoor("photo.jpg")
        >>> print(f"{result.label}: {result.confidence:.1%}")
    """

    def __init__(self, model_variant: str = "so400m", device: str | None = None):
        """Initialize the classifier.

        Args:
            model_variant: SigLIP model variant (base, so400m, large-v2, etc.)
            device: Device to run on (None for auto-detect)
        """
        self._model_variant = model_variant
        self._device = device
        self._model = None
        self._processor = None
        self._torch_dtype = None

    def _load_model(self) -> None:
        """Lazy load the SigLIP model."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "SigLIP dependencies not installed. Install with:\n"
                "pip install torch torchvision transformers>=4.47.0"
            ) from e

        # Model variants
        model_map = {
            "base": "google/siglip-base-patch16-224",
            "so400m": "google/siglip-so400m-patch14-384",
            "base-v2": "google/siglip2-base-patch16-224",
            "large-v2": "google/siglip2-large-patch16-512",
            "so400m-v2": "google/siglip2-so400m-patch14-384",
        }
        model_id = model_map.get(self._model_variant, model_map["so400m"])

        # Determine device
        if self._device:
            device = torch.device(self._device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Determine dtype
        if device.type == "cuda":
            if torch.cuda.get_device_capability()[0] >= 8:
                self._torch_dtype = torch.bfloat16
            else:
                self._torch_dtype = torch.float16
        else:
            self._torch_dtype = torch.float32

        logger.info(f"Loading SigLIP classifier ({model_id}) on {device}")

        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=self._torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self._model.eval()
        self._device = device

    def classify(
        self,
        image_path: Path | str,
        categories: dict[str, list[str]],
    ) -> ClassificationResult:
        """Classify image into one of the given categories.

        Uses prompt ensembling for better accuracy.

        Args:
            image_path: Path to image file
            categories: Dict mapping label to list of prompt templates
                        e.g., {"indoor": ["an indoor photo", ...], ...}

        Returns:
            ClassificationResult with best label and all scores
        """
        import torch
        from PIL import Image

        self._load_model()

        image = Image.open(image_path).convert("RGB")

        # Flatten all prompts
        all_prompts = []
        prompt_to_label = {}
        for label, prompts in categories.items():
            for prompt in prompts:
                all_prompts.append(prompt.lower())
                prompt_to_label[prompt.lower()] = label

        # Process inputs
        inputs = self._processor(
            text=all_prompts,
            images=image,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )

        # Move to device
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._device)
            if hasattr(inputs, "pixel_values"):
                inputs.pixel_values = inputs.pixel_values.to(self._torch_dtype)
        else:
            inputs = {k: v.to(self._device) if hasattr(v, "to") else v
                     for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self._torch_dtype)

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Get probabilities
        if hasattr(outputs, "logits_per_image"):
            logits = outputs.logits_per_image
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        probs = torch.sigmoid(logits).squeeze(0)
        probs_list = probs.cpu().float().numpy().tolist()

        # Average scores by label (ensemble)
        label_scores: dict[str, list[float]] = {}
        for prompt, prob in zip(all_prompts, probs_list, strict=True):
            label = prompt_to_label[prompt]
            if label not in label_scores:
                label_scores[label] = []
            label_scores[label].append(prob)

        # Compute average per label
        avg_scores = {
            label: sum(scores) / len(scores)
            for label, scores in label_scores.items()
        }

        # Find best label
        best_label = max(avg_scores, key=lambda k: avg_scores[k])

        return ClassificationResult(
            label=best_label,
            confidence=avg_scores[best_label],
            all_scores=avg_scores,
        )

    def classify_indoor_outdoor(self, image_path: Path | str) -> ClassificationResult:
        """Classify image as indoor or outdoor.

        Args:
            image_path: Path to image file

        Returns:
            ClassificationResult with indoor/outdoor label
        """
        return self.classify(image_path, SCENE_CATEGORIES["indoor_outdoor"])

    def classify_time_of_day(self, image_path: Path | str) -> ClassificationResult:
        """Classify time of day in image.

        Args:
            image_path: Path to image file

        Returns:
            ClassificationResult with time of day label
        """
        return self.classify(image_path, SCENE_CATEGORIES["time_of_day"])

    def classify_photo_type(self, image_path: Path | str) -> ClassificationResult:
        """Classify photo type (portrait, landscape, etc.).

        Args:
            image_path: Path to image file

        Returns:
            ClassificationResult with photo type label
        """
        return self.classify(image_path, SCENE_CATEGORIES["photo_type"])


def classify_scene(
    image_path: Path | str,
    category: str = "indoor_outdoor",
    model_variant: str = "so400m",
) -> ClassificationResult:
    """Convenience function for scene classification.

    Args:
        image_path: Path to image file
        category: Category to classify ("indoor_outdoor", "time_of_day", "photo_type")
        model_variant: SigLIP model to use

    Returns:
        ClassificationResult
    """
    classifier = SceneClassifier(model_variant=model_variant)

    if category == "indoor_outdoor":
        return classifier.classify_indoor_outdoor(image_path)
    elif category == "time_of_day":
        return classifier.classify_time_of_day(image_path)
    elif category == "photo_type":
        return classifier.classify_photo_type(image_path)
    else:
        categories = SCENE_CATEGORIES.get(category)
        if categories is None:
            raise ValueError(f"Unknown category: {category}")
        return classifier.classify(image_path, categories)
