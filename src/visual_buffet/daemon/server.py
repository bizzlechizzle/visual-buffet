"""Unix socket server for visual-buffet daemon.

Provides persistent daemon with Unix socket API for keeping ML models
warm between requests. Supports both standalone and orchestrator-managed modes.
"""

import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

from .protocol import MessageType, parse_message, serialize_message
from .model_pool import ModelPool

logger = logging.getLogger(__name__)

# Limits
MAX_IN_FLIGHT = 10  # Maximum concurrent requests

# Allowed image extensions for validation
ALLOWED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp',
    '.tiff', '.tif', '.heic', '.heif', '.raw',
    '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2',
}


class DaemonServer:
    """Unix socket server for visual-buffet daemon.

    Keeps ML models warm in VRAM and processes tagging requests via
    Unix domain socket. Supports graceful shutdown and health monitoring.

    Example:
        >>> server = DaemonServer("/tmp/visual-buffet.sock")
        >>> asyncio.run(server.start())
    """

    def __init__(
        self,
        socket_path: str,
        max_vram_percent: float = 0.80,
        managed: bool = False,
    ):
        """Initialize daemon server.

        Args:
            socket_path: Path for Unix domain socket
            max_vram_percent: Maximum VRAM usage for model pool
            managed: If True, run in managed mode for orchestrators
        """
        self.socket_path = Path(socket_path)
        self.managed = managed
        self.model_pool = ModelPool(max_vram_percent)

        # State
        self.engine = None
        self.server: asyncio.Server | None = None
        self.shutdown_event = asyncio.Event()
        self.in_flight: set[str] = set()
        self.request_count = 0
        self.error_count = 0
        self.start_time = 0.0

        # Progress socket for managed mode
        self.progress_socket_path = os.environ.get("PROGRESS_SOCKET")

    async def start(self) -> None:
        """Start the daemon server.

        Initializes the tagging engine, starts listening on Unix socket,
        and enters the main event loop.
        """
        # Cleanup stale socket
        if self.socket_path.exists():
            logger.info(f"Removing stale socket: {self.socket_path}")
            self.socket_path.unlink()

        # Initialize engine (lazy - models load on first use)
        logger.info("Initializing TaggingEngine...")
        from ..core.engine import TaggingEngine

        self.engine = TaggingEngine()
        self.start_time = time.time()

        # Start Unix socket server
        self.server = await asyncio.start_unix_server(
            self.handle_client,
            path=str(self.socket_path),
        )

        # Set permissions (owner only)
        os.chmod(self.socket_path, 0o600)

        # Write PID file
        pid_path = self.socket_path.with_suffix(".pid")
        pid_path.write_text(str(os.getpid()))

        logger.info(f"Daemon listening on {self.socket_path}")
        logger.info(f"Available plugins: {list(self.engine.plugins.keys())}")

        # Enter event loop
        async with self.server:
            await self.shutdown_event.wait()

        # Cleanup PID file
        pid_path.unlink(missing_ok=True)

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a client connection.

        Args:
            reader: Async stream reader
            writer: Async stream writer
        """
        client_addr = writer.get_extra_info("peername") or "unknown"
        logger.debug(f"Client connected: {client_addr}")

        # Send ready message
        try:
            from .. import __version__
        except ImportError:
            __version__ = "unknown"

        await self._send(
            writer,
            {
                "type": MessageType.READY.value,
                "version": __version__,
                "plugins": list(self.engine.plugins.keys()),
            },
        )

        try:
            while not self.shutdown_event.is_set():
                # Read with timeout for keepalive
                try:
                    data = await asyncio.wait_for(reader.readline(), timeout=60.0)
                except asyncio.TimeoutError:
                    # Send ping/keepalive
                    continue

                if not data:
                    # Client disconnected
                    break

                try:
                    msg = parse_message(data)
                    await self._handle_message(msg, writer)
                except json.JSONDecodeError as e:
                    await self._send_error(writer, None, "INVALID_JSON", str(e))

        except ConnectionResetError:
            logger.debug(f"Client disconnected: {client_addr}")
        except Exception as e:
            logger.exception(f"Client error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            logger.debug(f"Client connection closed: {client_addr}")

    async def _handle_message(
        self,
        msg: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        """Route message to appropriate handler.

        Args:
            msg: Parsed message dict
            writer: Response stream
        """
        msg_type = msg.get("type", "")

        handlers = {
            MessageType.TAG.value: self._handle_tag,
            MessageType.TAG_BATCH.value: self._handle_tag_batch,
            MessageType.HEALTH.value: self._handle_health,
            MessageType.MODELS.value: self._handle_models,
            MessageType.SHUTDOWN.value: self._handle_shutdown,
        }

        handler = handlers.get(msg_type)
        if handler:
            await handler(msg, writer)
        else:
            await self._send_error(
                writer,
                msg.get("request_id"),
                "UNKNOWN_MESSAGE_TYPE",
                f"Unknown message type: {msg_type}",
            )

    def _validate_image_path(self, path_str: str) -> Path:
        """Validate image path is safe and accessible.

        Args:
            path_str: Path string from request

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path is invalid or unsafe
        """
        path = Path(path_str).resolve()

        # Ensure it's a file
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {path}")

        # Check file extension
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        return path

    async def _handle_tag(
        self,
        msg: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle single image tag request.

        Args:
            msg: Tag request message
            writer: Response stream
        """
        request_id = msg.get("request_id", "")
        path_str = msg.get("path", "")

        if not path_str:
            await self._send_error(writer, request_id, "MISSING_PATH", "path is required")
            return

        # Check in-flight limit
        if len(self.in_flight) >= MAX_IN_FLIGHT:
            await self._send_error(
                writer, request_id,
                "TOO_MANY_REQUESTS",
                f"Server at capacity ({MAX_IN_FLIGHT} in-flight requests)",
            )
            return

        # Validate path
        try:
            path = self._validate_image_path(path_str)
        except ValueError as e:
            await self._send_error(writer, request_id, "INVALID_PATH", str(e), path_str)
            return

        self.in_flight.add(request_id)

        try:
            # Run tagging
            result = self.engine.tag_image(
                path,
                plugin_names=msg.get("plugins"),
                plugin_configs=msg.get("plugin_configs"),
                size=msg.get("size", "small"),
                save_tags=msg.get("save_tags", True),
            )

            # Write XMP if requested
            xmp_written = False
            if msg.get("write_xmp", True):
                try:
                    from ..services.xmp_handler import XMPHandler

                    xmp = XMPHandler()
                    xmp.write(path, result.get("results", {}))
                    xmp_written = True
                except Exception as e:
                    logger.warning(f"Failed to write XMP for {path}: {e}")

            # Send result
            await self._send(
                writer,
                {
                    "type": MessageType.RESULT.value,
                    "request_id": request_id,
                    "status": "success",
                    "path": str(path),
                    "results": result.get("results", {}),
                    "xmp_written": xmp_written,
                },
            )

            self.request_count += 1

        except FileNotFoundError:
            await self._send_error(
                writer, request_id, "FILE_NOT_FOUND", f"Image file not found: {path}", str(path)
            )
            self.error_count += 1
        except Exception as e:
            logger.exception(f"Tagging failed for {path}")
            await self._send_error(
                writer, request_id, "INFERENCE_FAILED", str(e), str(path)
            )
            self.error_count += 1
        finally:
            self.in_flight.discard(request_id)

    async def _handle_tag_batch(
        self,
        msg: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle batch tag request.

        Args:
            msg: Batch request message
            writer: Response stream
        """
        request_id = msg.get("request_id", "")
        paths = msg.get("paths", [])

        if not paths:
            await self._send_error(writer, request_id, "MISSING_PATHS", "paths is required")
            return

        self.in_flight.add(request_id)
        results = []
        total = len(paths)

        try:
            for i, path_str in enumerate(paths):
                path = Path(path_str)

                # Send progress
                await self._send(
                    writer,
                    {
                        "type": MessageType.PROGRESS.value,
                        "request_id": request_id,
                        "current": i,
                        "total": total,
                        "current_file": path.name,
                        "percent": (i / total) * 100,
                    },
                )

                try:
                    result = self.engine.tag_image(
                        path,
                        plugin_names=msg.get("plugins"),
                        size=msg.get("size", "small"),
                        save_tags=msg.get("save_tags", True),
                    )
                    results.append({"path": path_str, "status": "success", **result})
                except Exception as e:
                    results.append({"path": path_str, "status": "error", "error": str(e)})

            # Send final result
            await self._send(
                writer,
                {
                    "type": MessageType.RESULT.value,
                    "request_id": request_id,
                    "status": "success",
                    "total": total,
                    "successful": sum(1 for r in results if r["status"] == "success"),
                    "results": results,
                },
            )

            self.request_count += total

        finally:
            self.in_flight.discard(request_id)

    async def _handle_health(
        self,
        msg: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle health check request.

        Args:
            msg: Health request message
            writer: Response stream
        """
        # Get VRAM info
        vram_used = 0
        vram_total = 0
        try:
            import torch

            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() // (1024 * 1024)
                vram_total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        except ImportError:
            pass

        await self._send(
            writer,
            {
                "type": MessageType.HEALTH.value,
                "status": "healthy",
                "uptime_seconds": round(time.time() - self.start_time, 1),
                "requests_processed": self.request_count,
                "requests_failed": self.error_count,
                "models_loaded": list(self.model_pool.models.keys()),
                "vram_used_mb": vram_used,
                "vram_total_mb": vram_total,
                "in_flight": len(self.in_flight),
                "queue_depth": 0,
            },
        )

    async def _handle_models(
        self,
        msg: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle models status request.

        Args:
            msg: Models request message
            writer: Response stream
        """
        pool_status = self.model_pool.get_status()

        available = [
            name
            for name, plugin in self.engine.plugins.items()
            if name not in self.model_pool and plugin.is_available()
        ]

        await self._send(
            writer,
            {
                "type": MessageType.MODELS.value,
                "loaded": pool_status["models_loaded"],
                "available": available,
                "vram_used_mb": pool_status["vram_used_mb"],
                "vram_max_mb": pool_status["vram_max_mb"],
            },
        )

    async def _handle_shutdown(
        self,
        msg: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle shutdown request.

        Args:
            msg: Shutdown request message
            writer: Response stream
        """
        reason = msg.get("reason", "Requested")
        logger.info(f"Shutdown requested: {reason}")

        await self._send(
            writer,
            {"type": MessageType.ACK.value, "command": "shutdown"},
        )

        # Trigger shutdown
        asyncio.create_task(self.shutdown())

    async def _send(self, writer: asyncio.StreamWriter, msg: dict) -> None:
        """Send message to client.

        Args:
            writer: Stream writer
            msg: Message dict to send
        """
        writer.write(serialize_message(msg))
        await writer.drain()

    async def _send_error(
        self,
        writer: asyncio.StreamWriter,
        request_id: str | None,
        code: str,
        message: str,
        path: str | None = None,
    ) -> None:
        """Send error response.

        Args:
            writer: Stream writer
            request_id: Original request ID
            code: Error code
            message: Error message
            path: Related file path
        """
        error = {
            "type": MessageType.ERROR.value,
            "request_id": request_id,
            "code": code,
            "message": message,
        }
        if path:
            error["path"] = path

        await self._send(writer, error)

    async def shutdown(self) -> None:
        """Graceful shutdown.

        Waits for in-flight requests, unloads models, cleans up socket.
        """
        logger.info("Graceful shutdown initiated")

        # Wait for in-flight requests (max 30 seconds)
        if self.in_flight:
            logger.info(f"Waiting for {len(self.in_flight)} in-flight requests")
            for _ in range(300):  # 30 second timeout
                if not self.in_flight:
                    break
                await asyncio.sleep(0.1)

            if self.in_flight:
                logger.warning(f"Timeout waiting for {len(self.in_flight)} requests")

        # Unload models
        await self.model_pool.unload_all()

        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Cleanup socket
        if self.socket_path.exists():
            self.socket_path.unlink()

        # Signal shutdown complete
        self.shutdown_event.set()
        logger.info("Shutdown complete")


def setup_signal_handlers(server: DaemonServer) -> None:
    """Set up signal handlers for graceful shutdown.

    Args:
        server: Daemon server instance
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop yet, signals will be set when loop starts
        return

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(
                _handle_signal(server, s)
            ),
        )


async def _handle_signal(server: DaemonServer, sig: signal.Signals) -> None:
    """Handle shutdown signal.

    Args:
        server: Daemon server instance
        sig: Signal received
    """
    logger.info(f"Received {sig.name}, initiating shutdown")
    await server.shutdown()
