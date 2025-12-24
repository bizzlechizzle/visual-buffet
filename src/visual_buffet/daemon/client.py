"""Client library for visual-buffet daemon.

Provides a convenient Python API for interacting with the daemon
via Unix domain socket.
"""

import asyncio
import json
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TagResult:
    """Result from a tag request."""

    request_id: str
    path: str
    status: str
    results: dict[str, Any]
    xmp_written: bool = False
    error: str | None = None


@dataclass
class HealthStatus:
    """Daemon health status."""

    status: str
    uptime_seconds: float
    requests_processed: int
    requests_failed: int
    models_loaded: list[str]
    vram_used_mb: int
    vram_total_mb: int
    in_flight: int
    queue_depth: int


class DaemonClient:
    """Synchronous client for visual-buffet daemon.

    Example:
        >>> client = DaemonClient("/tmp/visual-buffet.sock")
        >>> client.connect()
        >>> result = client.tag("/path/to/image.jpg")
        >>> print(result.results)
        >>> client.close()

    Context manager:
        >>> with DaemonClient() as client:
        ...     result = client.tag("/path/to/image.jpg")
    """

    def __init__(self, socket_path: str = "/tmp/visual-buffet.sock"):
        """Initialize client.

        Args:
            socket_path: Path to daemon Unix socket
        """
        self.socket_path = socket_path
        self._sock: socket.socket | None = None
        self._ready_info: dict[str, Any] = {}

    def connect(self, timeout: float = 10.0) -> dict[str, Any]:
        """Connect to daemon.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            Ready message from daemon with version and available plugins

        Raises:
            ConnectionError: If daemon is not available
        """
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)

        try:
            self._sock.connect(self.socket_path)
        except FileNotFoundError:
            raise ConnectionError(f"Daemon socket not found: {self.socket_path}")
        except ConnectionRefusedError:
            raise ConnectionError(f"Daemon not accepting connections: {self.socket_path}")

        # Read ready message
        self._ready_info = self._recv_message()
        if self._ready_info.get("type") != "ready":
            raise ConnectionError(f"Unexpected response: {self._ready_info}")

        return self._ready_info

    def close(self) -> None:
        """Close connection to daemon."""
        if self._sock:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> "DaemonClient":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    @property
    def version(self) -> str:
        """Get daemon version."""
        return self._ready_info.get("version", "unknown")

    @property
    def available_plugins(self) -> list[str]:
        """Get list of available plugins."""
        return self._ready_info.get("plugins", [])

    def _send_message(self, msg: dict[str, Any]) -> None:
        """Send message to daemon."""
        if not self._sock:
            raise ConnectionError("Not connected")
        data = (json.dumps(msg) + "\n").encode()
        self._sock.sendall(data)

    def _recv_message(self) -> dict[str, Any]:
        """Receive message from daemon."""
        if not self._sock:
            raise ConnectionError("Not connected")

        data = b""
        while not data.endswith(b"\n"):
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk

        return json.loads(data.decode().strip())

    def tag(
        self,
        path: str | Path,
        plugins: list[str] | None = None,
        size: str = "small",
        save_tags: bool = True,
        write_xmp: bool = True,
        timeout: float = 300.0,
    ) -> TagResult:
        """Tag a single image.

        Args:
            path: Path to image file
            plugins: List of plugin names to use (None = all)
            size: Image size (little/small/large/huge/original)
            save_tags: Whether to save tags JSON file
            write_xmp: Whether to write XMP sidecar
            timeout: Request timeout in seconds

        Returns:
            TagResult with tagging results
        """
        if not self._sock:
            raise ConnectionError("Not connected")

        request_id = str(uuid.uuid4())
        self._sock.settimeout(timeout)

        self._send_message({
            "type": "tag",
            "request_id": request_id,
            "path": str(path),
            "plugins": plugins,
            "size": size,
            "save_tags": save_tags,
            "write_xmp": write_xmp,
        })

        response = self._recv_message()

        if response.get("type") == "error":
            return TagResult(
                request_id=request_id,
                path=str(path),
                status="error",
                results={},
                error=response.get("message"),
            )

        return TagResult(
            request_id=response.get("request_id", request_id),
            path=response.get("path", str(path)),
            status=response.get("status", "unknown"),
            results=response.get("results", {}),
            xmp_written=response.get("xmp_written", False),
        )

    def tag_batch(
        self,
        paths: list[str | Path],
        plugins: list[str] | None = None,
        size: str = "small",
        timeout: float = 600.0,
        on_progress: callable | None = None,
    ) -> list[TagResult]:
        """Tag multiple images.

        Args:
            paths: List of image paths
            plugins: List of plugin names to use
            size: Image size
            timeout: Request timeout in seconds
            on_progress: Optional callback(current, total, current_file)

        Returns:
            List of TagResult objects
        """
        if not self._sock:
            raise ConnectionError("Not connected")

        request_id = str(uuid.uuid4())
        self._sock.settimeout(timeout)

        self._send_message({
            "type": "tag_batch",
            "request_id": request_id,
            "paths": [str(p) for p in paths],
            "plugins": plugins,
            "size": size,
        })

        # Read progress and final result
        while True:
            response = self._recv_message()

            if response.get("type") == "progress":
                if on_progress:
                    on_progress(
                        response.get("current", 0),
                        response.get("total", len(paths)),
                        response.get("current_file", ""),
                    )
                continue

            if response.get("type") == "result":
                results = []
                for r in response.get("results", []):
                    results.append(TagResult(
                        request_id=request_id,
                        path=r.get("path", ""),
                        status=r.get("status", "unknown"),
                        results=r.get("results", {}),
                        error=r.get("error"),
                    ))
                return results

            if response.get("type") == "error":
                raise RuntimeError(
                    f"Batch request failed: {response.get('message')}"
                )

    def health(self) -> HealthStatus:
        """Get daemon health status.

        Returns:
            HealthStatus with daemon metrics
        """
        if not self._sock:
            raise ConnectionError("Not connected")

        self._send_message({"type": "health"})
        response = self._recv_message()

        return HealthStatus(
            status=response.get("status", "unknown"),
            uptime_seconds=response.get("uptime_seconds", 0),
            requests_processed=response.get("requests_processed", 0),
            requests_failed=response.get("requests_failed", 0),
            models_loaded=response.get("models_loaded", []),
            vram_used_mb=response.get("vram_used_mb", 0),
            vram_total_mb=response.get("vram_total_mb", 0),
            in_flight=response.get("in_flight", 0),
            queue_depth=response.get("queue_depth", 0),
        )

    def models(self) -> dict[str, Any]:
        """Get model status.

        Returns:
            Dict with loaded and available models
        """
        if not self._sock:
            raise ConnectionError("Not connected")

        self._send_message({"type": "models"})
        return self._recv_message()

    def shutdown(self, reason: str = "Client requested") -> bool:
        """Request daemon shutdown.

        Args:
            reason: Reason for shutdown

        Returns:
            True if shutdown was acknowledged
        """
        if not self._sock:
            raise ConnectionError("Not connected")

        self._send_message({
            "type": "shutdown",
            "reason": reason,
        })

        response = self._recv_message()
        return response.get("type") == "ack"


class AsyncDaemonClient:
    """Async client for visual-buffet daemon.

    Example:
        >>> async with AsyncDaemonClient() as client:
        ...     result = await client.tag("/path/to/image.jpg")
    """

    def __init__(self, socket_path: str = "/tmp/visual-buffet.sock"):
        """Initialize async client.

        Args:
            socket_path: Path to daemon Unix socket
        """
        self.socket_path = socket_path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._ready_info: dict[str, Any] = {}

    async def connect(self, timeout: float = 10.0) -> dict[str, Any]:
        """Connect to daemon asynchronously.

        Returns:
            Ready message from daemon
        """
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self.socket_path),
                timeout=timeout,
            )
        except FileNotFoundError:
            raise ConnectionError(f"Daemon socket not found: {self.socket_path}")
        except ConnectionRefusedError:
            raise ConnectionError(f"Daemon not accepting connections")

        # Read ready message
        self._ready_info = await self._recv_message()
        if self._ready_info.get("type") != "ready":
            raise ConnectionError(f"Unexpected response: {self._ready_info}")

        return self._ready_info

    async def close(self) -> None:
        """Close connection."""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None
            self._reader = None

    async def __aenter__(self) -> "AsyncDaemonClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    @property
    def version(self) -> str:
        """Get daemon version."""
        return self._ready_info.get("version", "unknown")

    @property
    def available_plugins(self) -> list[str]:
        """Get list of available plugins."""
        return self._ready_info.get("plugins", [])

    async def _send_message(self, msg: dict[str, Any]) -> None:
        """Send message to daemon."""
        if not self._writer:
            raise ConnectionError("Not connected")
        data = (json.dumps(msg) + "\n").encode()
        self._writer.write(data)
        await self._writer.drain()

    async def _recv_message(self) -> dict[str, Any]:
        """Receive message from daemon."""
        if not self._reader:
            raise ConnectionError("Not connected")
        data = await self._reader.readline()
        if not data:
            raise ConnectionError("Connection closed")
        return json.loads(data.decode().strip())

    async def tag(
        self,
        path: str | Path,
        plugins: list[str] | None = None,
        size: str = "small",
        save_tags: bool = True,
        write_xmp: bool = True,
    ) -> TagResult:
        """Tag a single image asynchronously.

        Args:
            path: Path to image file
            plugins: List of plugin names to use
            size: Image size
            save_tags: Save tags JSON
            write_xmp: Write XMP sidecar

        Returns:
            TagResult
        """
        if not self._writer:
            raise ConnectionError("Not connected")

        request_id = str(uuid.uuid4())

        await self._send_message({
            "type": "tag",
            "request_id": request_id,
            "path": str(path),
            "plugins": plugins,
            "size": size,
            "save_tags": save_tags,
            "write_xmp": write_xmp,
        })

        response = await self._recv_message()

        if response.get("type") == "error":
            return TagResult(
                request_id=request_id,
                path=str(path),
                status="error",
                results={},
                error=response.get("message"),
            )

        return TagResult(
            request_id=response.get("request_id", request_id),
            path=response.get("path", str(path)),
            status=response.get("status", "unknown"),
            results=response.get("results", {}),
            xmp_written=response.get("xmp_written", False),
        )

    async def health(self) -> HealthStatus:
        """Get daemon health status asynchronously."""
        if not self._writer:
            raise ConnectionError("Not connected")

        await self._send_message({"type": "health"})
        response = await self._recv_message()

        return HealthStatus(
            status=response.get("status", "unknown"),
            uptime_seconds=response.get("uptime_seconds", 0),
            requests_processed=response.get("requests_processed", 0),
            requests_failed=response.get("requests_failed", 0),
            models_loaded=response.get("models_loaded", []),
            vram_used_mb=response.get("vram_used_mb", 0),
            vram_total_mb=response.get("vram_total_mb", 0),
            in_flight=response.get("in_flight", 0),
            queue_depth=response.get("queue_depth", 0),
        )

    async def shutdown(self, reason: str = "Client requested") -> bool:
        """Request daemon shutdown asynchronously."""
        if not self._writer:
            raise ConnectionError("Not connected")

        await self._send_message({
            "type": "shutdown",
            "reason": reason,
        })

        response = await self._recv_message()
        return response.get("type") == "ack"
