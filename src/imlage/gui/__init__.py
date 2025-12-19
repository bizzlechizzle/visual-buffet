"""Web GUI for IMLAGE.

Provides a FastAPI-based web interface for image tagging.
"""

from .server import app, create_app

__all__ = ["app", "create_app"]
