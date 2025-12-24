"""Visual-buffet daemon mode.

Persistent daemon with Unix socket API for keeping ML models warm.

Server:
    >>> from visual_buffet.daemon import DaemonServer
    >>> server = DaemonServer("/tmp/vb.sock")
    >>> asyncio.run(server.start())

Client (sync):
    >>> from visual_buffet.daemon import DaemonClient
    >>> with DaemonClient() as client:
    ...     result = client.tag("/path/to/image.jpg")

Client (async):
    >>> from visual_buffet.daemon import AsyncDaemonClient
    >>> async with AsyncDaemonClient() as client:
    ...     result = await client.tag("/path/to/image.jpg")
"""

from .server import DaemonServer
from .protocol import MessageType, TagRequest, parse_message, serialize_message
from .model_pool import ModelPool
from .client import DaemonClient, AsyncDaemonClient, TagResult, HealthStatus

__all__ = [
    # Server
    "DaemonServer",
    "ModelPool",
    # Protocol
    "MessageType",
    "TagRequest",
    "parse_message",
    "serialize_message",
    # Client
    "DaemonClient",
    "AsyncDaemonClient",
    "TagResult",
    "HealthStatus",
]
