"""
Progress Reporter Tests

Tests for Unix socket-based progress reporting.
"""

import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import importlib.util

# Direct file import to avoid package __init__ chain that loads heavy dependencies
_module_path = Path(__file__).parent.parent / "src" / "visual_buffet" / "core" / "progress_reporter.py"
_spec = importlib.util.spec_from_file_location("progress_reporter", _module_path)
_module = importlib.util.module_from_spec(_spec)
sys.modules["progress_reporter"] = _module
_spec.loader.exec_module(_module)

# Import from the directly loaded module
VISUAL_BUFFET_STAGES = _module.VISUAL_BUFFET_STAGES
ProgressData = _module.ProgressData
ProgressReporter = _module.ProgressReporter
get_progress_reporter = _module.get_progress_reporter


class TestVisualBuffetStages:
    """Test stage definitions."""

    def test_stage_weights_sum_to_100(self):
        """Stage weights should sum to 100."""
        total_weight = sum(s["weight"] for s in VISUAL_BUFFET_STAGES.values())
        assert total_weight == 100

    def test_all_stages_defined(self):
        """All expected stages should be defined."""
        assert "loading" in VISUAL_BUFFET_STAGES
        assert "tagging" in VISUAL_BUFFET_STAGES
        assert "writing" in VISUAL_BUFFET_STAGES

    def test_stages_have_sequential_numbers(self):
        """Stages should have sequential numbers."""
        for i, (name, stage) in enumerate(VISUAL_BUFFET_STAGES.items(), 1):
            assert stage["number"] == i
            assert stage["total_stages"] == 3


class TestProgressReporter:
    """Test ProgressReporter class."""

    @pytest.fixture
    def socket_server(self):
        """Create a mock socket server."""
        # Use /tmp directly for shorter socket paths (macOS has 104 char limit)
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="vb-", dir="/tmp")
        socket_path = os.path.join(tmp_dir, "p.sock")
        messages = []
        server_conn = {"socket": None}

        def accept_connection(server_socket):
            try:
                conn, _ = server_socket.accept()
                server_conn["socket"] = conn
                buffer = ""
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    buffer += data.decode()
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line:
                            messages.append(json.loads(line))
            except Exception:
                pass

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        server.listen(1)

        thread = threading.Thread(target=accept_connection, args=(server,))
        thread.daemon = True
        thread.start()

        yield {
            "path": socket_path,
            "messages": messages,
            "server_conn": server_conn,
            "server": server,
            "tmp_dir": tmp_dir,
        }

        server.close()
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    @pytest.fixture
    def reporter_with_socket(self, socket_server):
        """Create reporter connected to mock socket."""
        os.environ["PROGRESS_SOCKET"] = socket_server["path"]
        os.environ["PROGRESS_SESSION_ID"] = "test-session-789"

        reporter = ProgressReporter()
        connected = reporter.connect()
        time.sleep(0.05)  # Allow connection to establish

        yield reporter, socket_server

        reporter.close()
        del os.environ["PROGRESS_SOCKET"]
        del os.environ["PROGRESS_SESSION_ID"]

    def test_connect_with_socket(self, reporter_with_socket):
        """Should connect when PROGRESS_SOCKET is set."""
        reporter, _ = reporter_with_socket
        assert reporter.is_connected

    def test_connect_without_socket(self):
        """Should return False when PROGRESS_SOCKET not set."""
        if "PROGRESS_SOCKET" in os.environ:
            del os.environ["PROGRESS_SOCKET"]

        reporter = ProgressReporter()
        assert not reporter.connect()
        assert not reporter.is_connected

    def test_send_message_format(self, reporter_with_socket):
        """Messages should include required fields."""
        reporter, socket_server = reporter_with_socket

        reporter.send({"type": "test", "custom": "data"})
        time.sleep(0.1)

        assert len(socket_server["messages"]) >= 1
        msg = socket_server["messages"][-1]

        assert msg["type"] == "test"
        assert msg["custom"] == "data"
        assert "timestamp" in msg
        assert msg["session_id"] == "test-session-789"
        assert msg["app"] == "visual-buffet"
        assert "app_version" in msg

    def test_stage_started(self, reporter_with_socket):
        """Should send stage_started message."""
        reporter, socket_server = reporter_with_socket

        reporter.stage_started("tagging")
        time.sleep(0.1)

        msg = socket_server["messages"][-1]
        assert msg["type"] == "stage_started"
        assert msg["stage"]["name"] == "tagging"
        assert msg["stage"]["display_name"] == "Running inference"
        assert msg["stage"]["number"] == 2
        assert msg["stage"]["total_stages"] == 3

    def test_stage_completed(self, reporter_with_socket):
        """Should send stage_completed message."""
        reporter, socket_server = reporter_with_socket

        reporter.stage_completed("tagging", 5000, 100)
        time.sleep(0.1)

        msg = socket_server["messages"][-1]
        assert msg["type"] == "stage_completed"
        assert msg["stage"]["name"] == "tagging"
        assert msg["duration_ms"] == 5000
        assert msg["items_processed"] == 100

    def test_progress(self, reporter_with_socket):
        """Should send progress message with all fields."""
        reporter, socket_server = reporter_with_socket
        reporter.reset_start_time()

        data = ProgressData(
            completed=50,
            total=100,
            failed=2,
            skipped=3,
            current_file="/path/to/image.jpg",
            percent_complete=50.0,
            eta_ms=30000,
        )

        reporter.progress("tagging", data)
        time.sleep(0.1)

        msg = socket_server["messages"][-1]
        assert msg["type"] == "progress"
        assert msg["stage"]["name"] == "tagging"
        assert msg["stage"]["weight"] == 90
        assert msg["items"]["completed"] == 50
        assert msg["items"]["total"] == 100
        assert msg["items"]["failed"] == 2
        assert msg["items"]["skipped"] == 3
        assert msg["current"]["item"] == "/path/to/image.jpg"
        assert msg["current"]["item_short"] == "image.jpg"
        assert msg["timing"]["eta_ms"] == 30000
        assert msg["percent_complete"] == 50.0

    def test_complete_with_failures(self, reporter_with_socket):
        """Complete message with failures should have exit_code 1."""
        reporter, socket_server = reporter_with_socket

        reporter.complete(
            total_items=100,
            successful=95,
            failed=3,
            skipped=2,
            duration_ms=60000,
        )
        time.sleep(0.1)

        msg = socket_server["messages"][-1]
        assert msg["type"] == "complete"
        assert msg["summary"]["total_items"] == 100
        assert msg["summary"]["successful"] == 95
        assert msg["summary"]["failed"] == 3
        assert msg["summary"]["skipped"] == 2
        assert msg["summary"]["duration_ms"] == 60000
        assert msg["exit_code"] == 1

    def test_complete_without_failures(self, reporter_with_socket):
        """Complete message without failures should have exit_code 0."""
        reporter, socket_server = reporter_with_socket

        reporter.complete(
            total_items=100,
            successful=100,
            failed=0,
            skipped=0,
            duration_ms=60000,
        )
        time.sleep(0.1)

        msg = socket_server["messages"][-1]
        assert msg["exit_code"] == 0


class TestControlCommands:
    """Test control command handling."""

    @pytest.fixture
    def bidirectional_socket(self):
        """Create a bidirectional socket for testing control commands."""
        import tempfile
        import shutil
        tmp_dir = tempfile.mkdtemp(prefix="vb-", dir="/tmp")
        socket_path = os.path.join(tmp_dir, "p.sock")
        messages = []
        client_conn = {"socket": None}

        def accept_connection(server_socket):
            try:
                conn, _ = server_socket.accept()
                client_conn["socket"] = conn
                buffer = ""
                while True:
                    try:
                        data = conn.recv(4096)
                        if not data:
                            break
                        buffer += data.decode()
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            if line:
                                messages.append(json.loads(line))
                    except Exception:
                        break
            except Exception:
                pass

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        server.listen(1)

        thread = threading.Thread(target=accept_connection, args=(server,))
        thread.daemon = True
        thread.start()

        os.environ["PROGRESS_SOCKET"] = socket_path
        os.environ["PROGRESS_SESSION_ID"] = "test-ctrl"

        reporter = ProgressReporter()
        reporter.connect()
        time.sleep(0.05)

        yield {
            "reporter": reporter,
            "messages": messages,
            "client_conn": client_conn,
            "server": server,
            "tmp_dir": tmp_dir,
        }

        reporter.close()
        server.close()
        del os.environ["PROGRESS_SOCKET"]
        del os.environ["PROGRESS_SESSION_ID"]
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_pause_command(self, bidirectional_socket):
        """Should handle pause command."""
        reporter = bidirectional_socket["reporter"]
        client_conn = bidirectional_socket["client_conn"]

        pause_received = threading.Event()
        reporter.on("pause", lambda: pause_received.set())

        # Send pause command
        client_conn["socket"].sendall(
            (json.dumps({"type": "control", "command": "pause"}) + "\n").encode()
        )
        time.sleep(0.1)

        assert reporter.paused
        assert pause_received.is_set()

        # Check ack was sent
        ack_msgs = [m for m in bidirectional_socket["messages"]
                    if m.get("type") == "ack" and m.get("command") == "pause"]
        assert len(ack_msgs) > 0

    def test_resume_command(self, bidirectional_socket):
        """Should handle resume command."""
        reporter = bidirectional_socket["reporter"]
        client_conn = bidirectional_socket["client_conn"]

        # First pause
        client_conn["socket"].sendall(
            (json.dumps({"type": "control", "command": "pause"}) + "\n").encode()
        )
        time.sleep(0.1)
        assert reporter.paused

        # Then resume
        resume_received = threading.Event()
        reporter.on("resume", lambda: resume_received.set())

        client_conn["socket"].sendall(
            (json.dumps({"type": "control", "command": "resume"}) + "\n").encode()
        )
        time.sleep(0.1)

        assert not reporter.paused
        assert resume_received.is_set()

    def test_cancel_command(self, bidirectional_socket):
        """Should handle cancel command with reason."""
        reporter = bidirectional_socket["reporter"]
        client_conn = bidirectional_socket["client_conn"]

        cancel_reason = None

        def on_cancel(reason):
            nonlocal cancel_reason
            cancel_reason = reason

        reporter.on("cancel", on_cancel)

        client_conn["socket"].sendall(
            (json.dumps({
                "type": "control",
                "command": "cancel",
                "reason": "User requested"
            }) + "\n").encode()
        )
        time.sleep(0.1)

        assert reporter.cancelled
        assert cancel_reason == "User requested"
        assert not reporter.should_continue()

    def test_should_continue(self, bidirectional_socket):
        """should_continue should reflect cancelled state."""
        reporter = bidirectional_socket["reporter"]
        client_conn = bidirectional_socket["client_conn"]

        assert reporter.should_continue()

        client_conn["socket"].sendall(
            (json.dumps({"type": "control", "command": "cancel"}) + "\n").encode()
        )
        time.sleep(0.1)

        assert not reporter.should_continue()


class TestWaitWhilePaused:
    """Test wait_while_paused behavior."""

    def test_wait_while_paused_blocks(self):
        """wait_while_paused should block until resumed."""
        import tempfile
        import shutil
        tmp_dir = tempfile.mkdtemp(prefix="vb-", dir="/tmp")
        socket_path = os.path.join(tmp_dir, "p.sock")
        client_conn = {"socket": None}

        def accept_connection(server_socket):
            conn, _ = server_socket.accept()
            client_conn["socket"] = conn
            while True:
                try:
                    conn.recv(4096)
                except Exception:
                    break

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        server.listen(1)

        thread = threading.Thread(target=accept_connection, args=(server,))
        thread.daemon = True
        thread.start()

        os.environ["PROGRESS_SOCKET"] = socket_path
        reporter = ProgressReporter()
        reporter.connect()
        time.sleep(0.05)

        # Pause
        client_conn["socket"].sendall(
            (json.dumps({"type": "control", "command": "pause"}) + "\n").encode()
        )
        time.sleep(0.1)
        assert reporter.paused

        # Start waiting in background
        wait_completed = threading.Event()

        def wait_thread():
            reporter.wait_while_paused()
            wait_completed.set()

        t = threading.Thread(target=wait_thread)
        t.start()

        # Should still be waiting
        time.sleep(0.2)
        assert not wait_completed.is_set()

        # Resume
        client_conn["socket"].sendall(
            (json.dumps({"type": "control", "command": "resume"}) + "\n").encode()
        )

        # Should complete
        t.join(timeout=1)
        assert wait_completed.is_set()

        reporter.close()
        server.close()
        del os.environ["PROGRESS_SOCKET"]
        shutil.rmtree(tmp_dir, ignore_errors=True)


class TestStandaloneMode:
    """Test operation without socket."""

    def test_standalone_operation(self):
        """Should operate silently without socket."""
        if "PROGRESS_SOCKET" in os.environ:
            del os.environ["PROGRESS_SOCKET"]

        reporter = ProgressReporter()

        # Should not throw
        reporter.send({"type": "test"})
        reporter.stage_started("tagging")
        reporter.progress("tagging", ProgressData(
            completed=50,
            total=100,
            percent_complete=50.0,
        ))
        reporter.complete(
            total_items=100,
            successful=100,
            failed=0,
            skipped=0,
            duration_ms=5000,
        )

        assert not reporter.is_connected


class TestGetProgressReporter:
    """Test singleton behavior."""

    def test_returns_same_instance(self):
        """get_progress_reporter should return same instance."""
        # Note: This tests the module-level singleton
        reporter1 = get_progress_reporter()
        reporter2 = get_progress_reporter()

        assert reporter1 is reporter2
