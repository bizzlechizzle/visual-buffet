"""App-specific configuration for vocablearn databases.

Provides isolated database paths for each app, ensuring vocabulary
and OCR data remain separate across different use cases.

Example:
    >>> from vocablearn.app_config import AppConfig
    >>> config = AppConfig("abandoned-archive")
    >>> vocab = VocabLearn(config.vocab_db)
    >>> # Creates: ~/.abandoned-archive/data/vocabulary.db
"""

from pathlib import Path


class AppConfig:
    """App-specific configuration for database paths.

    Each app gets isolated storage in its own directory:
    - ~/.{app_name}/data/vocabulary.db - Tag vocabulary
    - ~/.{app_name}/data/ocr.db - OCR vocabulary
    - ~/.{app_name}/data/embeddings/ - Image embeddings

    Args:
        app_name: Unique app identifier (e.g., 'abandoned-archive')
        base_dir: Override base directory (default: ~/.{app_name})
    """

    def __init__(
        self,
        app_name: str,
        base_dir: Path | None = None,
    ):
        self.app_name = app_name
        self._base_dir = base_dir or Path.home() / f".{app_name}"
        self._data_dir = self._base_dir / "data"

    @property
    def base_dir(self) -> Path:
        """Base directory for all app data."""
        return self._base_dir

    @property
    def data_dir(self) -> Path:
        """Data directory (created on access)."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        return self._data_dir

    @property
    def vocab_db(self) -> Path:
        """Path to tag vocabulary database."""
        return self.data_dir / "vocabulary.db"

    @property
    def ocr_db(self) -> Path:
        """Path to OCR vocabulary database."""
        return self.data_dir / "ocr.db"

    @property
    def embeddings_dir(self) -> Path:
        """Directory for image embeddings."""
        path = self.data_dir / "embeddings"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_dir(self) -> Path:
        """Directory for cached data."""
        path = self._base_dir / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __repr__(self) -> str:
        return f"AppConfig(app_name={self.app_name!r}, base_dir={self._base_dir})"


# Pre-configured app instances
ABANDONED_ARCHIVE = AppConfig("abandoned-archive")
