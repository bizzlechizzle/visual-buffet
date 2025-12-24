"""Message protocol for Unix socket communication.

JSON-based protocol for visual-buffet daemon. All messages are
newline-delimited JSON objects.
"""

from dataclasses import dataclass, asdict
from enum import Enum
import json
from typing import Any


class MessageType(Enum):
    """Message types for daemon protocol."""

    # Requests (client -> server)
    TAG = "tag"
    TAG_BATCH = "tag_batch"
    HEALTH = "health"
    MODELS = "models"
    SHUTDOWN = "shutdown"

    # Responses (server -> client)
    READY = "ready"
    PROGRESS = "progress"
    RESULT = "result"
    ERROR = "error"
    ACK = "ack"


@dataclass
class TagRequest:
    """Request to tag a single image."""

    request_id: str
    path: str
    plugins: list[str] | None = None
    plugin_configs: dict[str, Any] | None = None
    size: str = "small"
    save_tags: bool = True
    write_xmp: bool = True


@dataclass
class TagBatchRequest:
    """Request to tag multiple images."""

    request_id: str
    paths: list[str]
    plugins: list[str] | None = None
    plugin_configs: dict[str, Any] | None = None
    size: str = "small"
    save_tags: bool = True
    write_xmp: bool = True


MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB


def parse_message(data: bytes) -> dict[str, Any]:
    """Parse incoming message from bytes.

    Args:
        data: Raw bytes from socket (newline-delimited JSON)

    Returns:
        Parsed message dict

    Raises:
        ValueError: If message exceeds size limit or is invalid JSON
    """
    if len(data) > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message exceeds maximum size ({MAX_MESSAGE_SIZE} bytes)")
    return json.loads(data.decode().strip())


def serialize_message(msg: dict[str, Any]) -> bytes:
    """Serialize outgoing message to bytes.

    Args:
        msg: Message dict to serialize

    Returns:
        Newline-delimited JSON bytes
    """
    return (json.dumps(msg) + "\n").encode()


def make_tag_request(
    path: str,
    request_id: str | None = None,
    plugins: list[str] | None = None,
    size: str = "small",
    save_tags: bool = True,
    write_xmp: bool = True,
) -> dict[str, Any]:
    """Create a tag request message.

    Args:
        path: Path to image file
        request_id: Optional request ID (generated if not provided)
        plugins: List of plugin names to use
        size: Image size for tagging (little/small/large/huge/original)
        save_tags: Whether to save tags JSON file
        write_xmp: Whether to write XMP sidecar

    Returns:
        Request message dict
    """
    import uuid

    return {
        "type": MessageType.TAG.value,
        "request_id": request_id or str(uuid.uuid4()),
        "path": path,
        "plugins": plugins,
        "size": size,
        "save_tags": save_tags,
        "write_xmp": write_xmp,
    }


def make_batch_request(
    paths: list[str],
    request_id: str | None = None,
    plugins: list[str] | None = None,
    size: str = "small",
) -> dict[str, Any]:
    """Create a batch tag request message.

    Args:
        paths: List of image file paths
        request_id: Optional request ID
        plugins: List of plugin names to use
        size: Image size for tagging

    Returns:
        Batch request message dict
    """
    import uuid

    return {
        "type": MessageType.TAG_BATCH.value,
        "request_id": request_id or str(uuid.uuid4()),
        "paths": paths,
        "plugins": plugins,
        "size": size,
    }


def make_health_request() -> dict[str, Any]:
    """Create a health check request."""
    return {"type": MessageType.HEALTH.value}


def make_shutdown_request(reason: str = "Requested") -> dict[str, Any]:
    """Create a shutdown request."""
    return {"type": MessageType.SHUTDOWN.value, "reason": reason}
