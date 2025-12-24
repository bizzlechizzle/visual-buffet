"""CLI progress tracking with EWMA-smoothed ETA.

Implements progress indicators following CLI UX best practices:
- EWMA-smoothed throughput for stable ETA estimates
- Human-readable time formatting
- Progress state tracking for batch operations

Based on: cli-progress-tracking.md SME document
"""

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class ProgressState:
    """Tracks progress for batch operations with EWMA-smoothed ETA."""

    # File progress
    files_total: int = 0
    files_processed: int = 0
    current_file: str = ""

    # Timing
    started_at: float = field(default_factory=perf_counter)

    # Throughput (EWMA smoothed)
    throughput_files_per_sec: float = 0.0
    _last_update_time: float = field(default_factory=perf_counter)
    _alpha: float = 0.15  # EWMA smoothing factor (0.1-0.2 recommended)

    # Track per-file timing for EWMA
    _file_times: list[float] = field(default_factory=list)

    def update(self, file_name: str, inference_time_ms: float) -> None:
        """Update progress after completing a file.

        Args:
            file_name: Name of completed file
            inference_time_ms: Time taken to process this file
        """
        now = perf_counter()
        self.files_processed += 1
        self.current_file = file_name

        # Calculate instantaneous throughput (files/sec)
        file_time_sec = inference_time_ms / 1000.0 if inference_time_ms > 0 else 0.001
        instant_throughput = 1.0 / file_time_sec

        # EWMA smooth throughput
        if self.throughput_files_per_sec == 0:
            # First sample - use directly
            self.throughput_files_per_sec = instant_throughput
        else:
            # Apply EWMA: ewma[n] = alpha * sample + (1 - alpha) * ewma[n-1]
            self.throughput_files_per_sec = (
                self._alpha * instant_throughput
                + (1 - self._alpha) * self.throughput_files_per_sec
            )

        self._last_update_time = now
        self._file_times.append(inference_time_ms)

    @property
    def files_remaining(self) -> int:
        """Number of files left to process."""
        return self.files_total - self.files_processed

    @property
    def percent_complete(self) -> float:
        """Progress as percentage (0-100)."""
        if self.files_total == 0:
            return 0.0
        return (self.files_processed / self.files_total) * 100

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return (perf_counter() - self.started_at) * 1000

    @property
    def eta_ms(self) -> float | None:
        """Estimated time remaining in milliseconds.

        Returns None if not enough data to estimate.
        """
        if self.files_processed == 0 or self.throughput_files_per_sec <= 0:
            return None

        remaining = self.files_remaining
        if remaining <= 0:
            return 0.0

        # ETA = remaining files / throughput (files/sec) * 1000 (to ms)
        eta_sec = remaining / self.throughput_files_per_sec
        return eta_sec * 1000

    @property
    def avg_time_per_file_ms(self) -> float:
        """Average processing time per file in ms."""
        if not self._file_times:
            return 0.0
        return sum(self._file_times) / len(self._file_times)


def format_duration(ms: float, style: str = "short") -> str:
    """Format duration in human-readable form.

    Args:
        ms: Duration in milliseconds
        style: 'short' (2m30s) or 'long' (2 minutes and 30 seconds)

    Returns:
        Formatted duration string

    Examples:
        >>> format_duration(150000)
        '2m30s'
        >>> format_duration(45000)
        '45s'
        >>> format_duration(500)
        '< 1s'
    """
    if ms < 0:
        return "now" if style == "short" else "now"

    if ms < 1000:
        return "< 1s" if style == "short" else "less than 1 second"

    total_seconds = int(ms / 1000)
    seconds = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = (total_seconds // 3600) % 24
    days = total_seconds // 86400

    if style == "short":
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        # Hide seconds for long durations (> 1 hour)
        if seconds > 0 and hours == 0:
            parts.append(f"{seconds}s")
        return "".join(parts) or "< 1s"

    # Long format with proper grammar
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    # Hide seconds for long durations
    if seconds > 0 and hours == 0:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    if not parts:
        return "less than 1 second"
    if len(parts) == 1:
        return parts[0]

    last = parts.pop()
    return f"{', '.join(parts)} and {last}"


def format_throughput(files_per_sec: float) -> str:
    """Format throughput as human-readable string.

    Args:
        files_per_sec: Processing speed in files per second

    Returns:
        Formatted string like "2.5 files/s" or "0.3 files/s"
    """
    if files_per_sec >= 1.0:
        return f"{files_per_sec:.1f} files/s"
    elif files_per_sec > 0:
        # Show as time per file for slow processing
        sec_per_file = 1.0 / files_per_sec
        if sec_per_file < 60:
            return f"{sec_per_file:.1f}s/file"
        else:
            return f"{sec_per_file / 60:.1f}m/file"
    else:
        return "-- files/s"


def truncate_middle(text: str, max_len: int) -> str:
    """Truncate text in the middle, preserving start and end.

    Args:
        text: Text to truncate
        max_len: Maximum length

    Returns:
        Truncated text with ... in middle if needed

    Example:
        >>> truncate_middle("very_long_filename.jpg", 15)
        'very_l...me.jpg'
    """
    if len(text) <= max_len:
        return text
    if max_len < 5:
        return text[:max_len]

    # Keep more of the end (extension is important)
    keep_end = min(max_len // 2, 10)
    keep_start = max_len - keep_end - 3  # 3 for "..."

    return f"{text[:keep_start]}...{text[-keep_end:]}"
