"""SigLIP embeddings for image similarity and duplicate detection.

Uses SigLIP image embeddings for finding similar/duplicate images
instead of tag-based Jaccard similarity.

Example:
    >>> index = EmbeddingIndex()
    >>> index.add_image("img001.jpg")
    >>> index.add_image("img002.jpg")
    >>> duplicates = index.find_duplicates(threshold=0.95)
"""

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Result of similarity search."""

    image_id: str
    score: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "image_id": self.image_id,
            "score": round(self.score, 4),
        }


@dataclass
class DuplicatePair:
    """A pair of potentially duplicate images."""

    image_a: str
    image_b: str
    similarity: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "image_a": self.image_a,
            "image_b": self.image_b,
            "similarity": round(self.similarity, 4),
        }


def get_image_embedding(
    image_path: Path | str,
    model_variant: str = "so400m",
    device: Optional[str] = None,
) -> list[float]:
    """Get SigLIP image embedding for an image.

    Args:
        image_path: Path to image file
        model_variant: SigLIP model variant
        device: Device to run on (None for auto)

    Returns:
        Embedding vector as list of floats
    """
    try:
        import torch
        from PIL import Image
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
    model_id = model_map.get(model_variant, model_map["so400m"])

    # Determine device
    if device:
        device_obj = torch.device(device)
    elif torch.cuda.is_available():
        device_obj = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_obj = torch.device("mps")
    else:
        device_obj = torch.device("cpu")

    # Determine dtype
    if device_obj.type == "cuda":
        if torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Load model
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_obj,
        trust_remote_code=True,
    )
    model.eval()

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Move to device
    if hasattr(inputs, "to"):
        inputs = inputs.to(device_obj)
        if hasattr(inputs, "pixel_values"):
            inputs.pixel_values = inputs.pixel_values.to(torch_dtype)
    else:
        inputs = {k: v.to(device_obj) if hasattr(v, "to") else v
                 for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch_dtype)

    # Get image embedding
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # Normalize embedding
    embedding = outputs[0].cpu().float()
    embedding = embedding / embedding.norm()

    return embedding.numpy().tolist()


class EmbeddingIndex:
    """Index for fast image embedding similarity search.

    Uses cosine similarity between SigLIP image embeddings
    for finding similar/duplicate images.

    Example:
        >>> index = EmbeddingIndex()
        >>> index.add_image("img001.jpg")
        >>> index.add_image("img002.jpg")
        >>> similar = index.find_similar("img001.jpg", top_k=5)
    """

    def __init__(
        self,
        model_variant: str = "so400m",
        device: Optional[str] = None,
    ):
        """Initialize the index.

        Args:
            model_variant: SigLIP model variant
            device: Device to run on (None for auto)
        """
        self._model_variant = model_variant
        self._device = device
        self._model = None
        self._processor = None
        self._torch_dtype = None

        # Index storage
        self._embeddings: dict[str, list[float]] = {}

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
        import torch as _torch
        if self._device:
            device = _torch.device(self._device)
        elif _torch.cuda.is_available():
            device = _torch.device("cuda")
        elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
            device = _torch.device("mps")
        else:
            device = _torch.device("cpu")

        # Determine dtype
        if device.type == "cuda":
            if _torch.cuda.get_device_capability()[0] >= 8:
                self._torch_dtype = _torch.bfloat16
            else:
                self._torch_dtype = _torch.float16
        else:
            self._torch_dtype = _torch.float32

        logger.info(f"Loading SigLIP embedding model on {device}")

        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=self._torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self._model.eval()
        self._device_obj = device

    def _get_embedding(self, image_path: Path | str) -> list[float]:
        """Get embedding for a single image."""
        import torch
        from PIL import Image

        self._load_model()

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")

        # Move to device
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._device_obj)
            if hasattr(inputs, "pixel_values"):
                inputs.pixel_values = inputs.pixel_values.to(self._torch_dtype)
        else:
            inputs = {k: v.to(self._device_obj) if hasattr(v, "to") else v
                     for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self._torch_dtype)

        # Get image embedding
        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)

        # Normalize embedding
        embedding = outputs[0].cpu().float()
        embedding = embedding / embedding.norm()

        return embedding.numpy().tolist()

    def add_image(
        self,
        image_path: Path | str,
        image_id: Optional[str] = None,
    ) -> str:
        """Add an image to the index.

        Args:
            image_path: Path to image file
            image_id: Optional ID (defaults to path)

        Returns:
            Image ID used in index
        """
        path = Path(image_path)
        if image_id is None:
            image_id = str(path)

        embedding = self._get_embedding(path)
        self._embeddings[image_id] = embedding

        return image_id

    def add_embedding(
        self,
        image_id: str,
        embedding: list[float],
    ) -> None:
        """Add a pre-computed embedding to the index.

        Args:
            image_id: Image identifier
            embedding: Pre-computed embedding vector
        """
        self._embeddings[image_id] = embedding

    def get_embedding(self, image_id: str) -> Optional[list[float]]:
        """Get embedding for an image.

        Args:
            image_id: Image identifier

        Returns:
            Embedding vector or None if not found
        """
        return self._embeddings.get(image_id)

    def remove_image(self, image_id: str) -> bool:
        """Remove an image from the index.

        Args:
            image_id: Image identifier

        Returns:
            True if removed, False if not found
        """
        if image_id in self._embeddings:
            del self._embeddings[image_id]
            return True
        return False

    def find_similar(
        self,
        image_id: str,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[SimilarityResult]:
        """Find images similar to the given image.

        Args:
            image_id: Image to find similar images for
            top_k: Number of results to return
            min_score: Minimum similarity score

        Returns:
            List of SimilarityResult, sorted by score descending
        """
        query_embedding = self._embeddings.get(image_id)
        if query_embedding is None:
            return []

        results = []
        for other_id, other_embedding in self._embeddings.items():
            if other_id == image_id:
                continue

            score = self._cosine_similarity(query_embedding, other_embedding)
            if score >= min_score:
                results.append(SimilarityResult(
                    image_id=other_id,
                    score=score,
                ))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def find_duplicates(
        self,
        threshold: float = 0.95,
    ) -> list[DuplicatePair]:
        """Find all duplicate image pairs.

        Args:
            threshold: Minimum similarity to consider duplicate

        Returns:
            List of DuplicatePair, sorted by similarity descending
        """
        duplicates = []
        image_ids = list(self._embeddings.keys())

        for i, id_a in enumerate(image_ids):
            emb_a = self._embeddings[id_a]

            for id_b in image_ids[i+1:]:
                emb_b = self._embeddings[id_b]

                score = self._cosine_similarity(emb_a, emb_b)
                if score >= threshold:
                    duplicates.append(DuplicatePair(
                        image_a=id_a,
                        image_b=id_b,
                        similarity=score,
                    ))

        # Sort by similarity descending
        duplicates.sort(key=lambda x: x.similarity, reverse=True)

        return duplicates

    def _cosine_similarity(
        self,
        emb_a: list[float],
        emb_b: list[float],
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot = sum(a * b for a, b in zip(emb_a, emb_b, strict=True))
        norm_a = sum(a * a for a in emb_a) ** 0.5
        norm_b = sum(b * b for b in emb_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def save(self, path: Path | str) -> None:
        """Save index to file.

        Args:
            path: Output file path
        """
        import json

        with open(path, "w") as f:
            json.dump({
                "model_variant": self._model_variant,
                "embeddings": self._embeddings,
            }, f)

    def load(self, path: Path | str) -> int:
        """Load index from file.

        Args:
            path: Input file path

        Returns:
            Number of embeddings loaded
        """
        import json

        with open(path) as f:
            data = json.load(f)

        self._model_variant = data.get("model_variant", self._model_variant)
        self._embeddings = data.get("embeddings", {})

        return len(self._embeddings)

    @property
    def size(self) -> int:
        """Number of images in index."""
        return len(self._embeddings)
