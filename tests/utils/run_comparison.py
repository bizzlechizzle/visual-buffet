"""
Run Comparison Utility for Visual Buffet Testing

Compares tag results across multiple CLI runs to verify consistency
and detect drift. This is the core utility for answering:
"Do multiple runs with the same settings produce the same results?"

Usage:
    # From Python
    from tests.utils.run_comparison import RunComparator
    comparator = RunComparator()
    comparator.execute_runs("images/test.jpg", num_runs=5)
    report = comparator.compare()

    # From CLI
    python -m tests.utils.run_comparison images/test.jpg --runs 5 --output report.json
"""

import json
import subprocess
import hashlib
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TagStats:
    """Statistics for a single tag across runs."""
    label: str
    occurrence_count: int  # How many runs contained this tag
    total_runs: int
    occurrence_rate: float  # occurrence_count / total_runs
    confidences: list[float] = field(default_factory=list)
    confidence_mean: float | None = None
    confidence_std: float | None = None
    confidence_min: float | None = None
    confidence_max: float | None = None

    def compute_stats(self):
        """Compute confidence statistics."""
        if self.confidences:
            self.confidence_mean = statistics.mean(self.confidences)
            self.confidence_min = min(self.confidences)
            self.confidence_max = max(self.confidences)
            if len(self.confidences) > 1:
                self.confidence_std = statistics.stdev(self.confidences)
            else:
                self.confidence_std = 0.0
        self.occurrence_rate = self.occurrence_count / self.total_runs


@dataclass
class RunResult:
    """Result from a single CLI run."""
    run_id: int
    timestamp: str
    image_path: str
    image_hash: str
    cli_args: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    tags: list[dict] = field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_cli_output(cls, run_id: int, image_path: Path, args: list[str],
                        result: subprocess.CompletedProcess, duration_ms: float) -> "RunResult":
        """Create from subprocess result."""
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()[:16]

        tags = []
        error = None
        if result.returncode == 0:
            try:
                output = json.loads(result.stdout)
                # Handle both single image and batch output formats
                if isinstance(output, list):
                    output = output[0] if output else {}

                # Extract tags from results
                results = output.get("results", {})
                for plugin_name, plugin_result in results.items():
                    plugin_tags = plugin_result.get("tags", [])
                    for tag in plugin_tags:
                        tags.append({
                            "plugin": plugin_name,
                            "label": tag.get("label"),
                            "confidence": tag.get("confidence"),
                        })
            except json.JSONDecodeError as e:
                error = f"JSON parse error: {e}"
        else:
            error = result.stderr or f"Exit code {result.returncode}"

        return cls(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat(),
            image_path=str(image_path),
            image_hash=image_hash,
            cli_args=args,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_ms=duration_ms,
            tags=tags,
            error=error,
        )


@dataclass
class ComparisonReport:
    """Report comparing multiple runs."""
    image_path: str
    image_hash: str
    num_runs: int
    num_successful: int
    cli_args: list[str]

    # Timing stats
    duration_mean_ms: float
    duration_std_ms: float

    # Tag consistency
    total_unique_tags: int
    tags_in_all_runs: int  # 100% occurrence
    tags_in_most_runs: int  # >80% occurrence
    tags_in_some_runs: int  # >50% occurrence

    # Overall consistency score (0-1)
    consistency_score: float

    # Per-tag statistics
    tag_stats: list[TagStats] = field(default_factory=list)

    # Individual run results
    runs: list[RunResult] = field(default_factory=list)

    # Flags
    is_deterministic: bool = False  # All runs identical
    has_high_variance_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["tag_stats"] = [asdict(ts) for ts in self.tag_stats]
        d["runs"] = [asdict(r) for r in self.runs]
        return d

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Run Comparison Report",
            f"=" * 50,
            f"Image: {self.image_path}",
            f"Hash: {self.image_hash}",
            f"Runs: {self.num_successful}/{self.num_runs} successful",
            f"",
            f"Timing:",
            f"  Mean: {self.duration_mean_ms:.1f}ms",
            f"  Std:  {self.duration_std_ms:.1f}ms",
            f"",
            f"Tag Consistency:",
            f"  Unique tags seen: {self.total_unique_tags}",
            f"  In ALL runs (100%): {self.tags_in_all_runs}",
            f"  In MOST runs (>80%): {self.tags_in_most_runs}",
            f"  In SOME runs (>50%): {self.tags_in_some_runs}",
            f"",
            f"Consistency Score: {self.consistency_score:.2%}",
            f"Deterministic: {'YES' if self.is_deterministic else 'NO'}",
        ]

        if self.has_high_variance_tags:
            lines.append(f"")
            lines.append(f"High Variance Tags (std > 0.1):")
            for tag in self.has_high_variance_tags:
                lines.append(f"  - {tag}")

        lines.append(f"")
        lines.append(f"Top 10 Most Consistent Tags:")
        sorted_tags = sorted(self.tag_stats, key=lambda t: t.occurrence_rate, reverse=True)
        for ts in sorted_tags[:10]:
            conf_str = f"{ts.confidence_mean:.3f}" if ts.confidence_mean else "N/A"
            std_str = f"±{ts.confidence_std:.3f}" if ts.confidence_std else ""
            lines.append(f"  {ts.label}: {ts.occurrence_rate:.0%} ({conf_str}{std_str})")

        return "\n".join(lines)


class RunComparator:
    """
    Execute and compare multiple CLI runs.

    Example:
        comparator = RunComparator(cli_command="visual-buffet")
        comparator.execute_runs("test.jpg", num_runs=5, plugins=["ram_plus"])
        report = comparator.compare()
        print(report.summary())
    """

    def __init__(self, cli_command: str = "visual-buffet"):
        self.cli_command = cli_command
        self.runs: list[RunResult] = []
        self.image_path: Path | None = None
        self.cli_args: list[str] = []

    def execute_runs(
        self,
        image_path: str | Path,
        num_runs: int = 5,
        plugins: list[str] | None = None,
        threshold: float | None = None,
        extra_args: list[str] | None = None,
    ) -> list[RunResult]:
        """
        Execute the CLI multiple times with same settings.

        Args:
            image_path: Path to image to tag
            num_runs: Number of times to run
            plugins: List of plugins to use (None = all)
            threshold: Confidence threshold
            extra_args: Additional CLI arguments

        Returns:
            List of RunResult objects
        """
        import time

        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        # Build CLI args
        args = ["tag", str(self.image_path), "--output", "-"]  # Output to stdout as JSON

        if plugins:
            for p in plugins:
                args.extend(["--plugin", p])

        if threshold is not None:
            args.extend(["--threshold", str(threshold)])

        if extra_args:
            args.extend(extra_args)

        self.cli_args = args
        self.runs = []

        for i in range(num_runs):
            start = time.perf_counter()
            result = subprocess.run(
                [self.cli_command] + args,
                capture_output=True,
                text=True,
            )
            duration_ms = (time.perf_counter() - start) * 1000

            run_result = RunResult.from_cli_output(
                run_id=i + 1,
                image_path=self.image_path,
                args=args,
                result=result,
                duration_ms=duration_ms,
            )
            self.runs.append(run_result)

        return self.runs

    def compare(self) -> ComparisonReport:
        """
        Compare all executed runs and generate report.

        Returns:
            ComparisonReport with statistics and consistency metrics
        """
        if not self.runs:
            raise ValueError("No runs to compare. Call execute_runs() first.")

        successful_runs = [r for r in self.runs if r.exit_code == 0]
        num_successful = len(successful_runs)

        # Timing statistics
        durations = [r.duration_ms for r in self.runs]
        duration_mean = statistics.mean(durations)
        duration_std = statistics.stdev(durations) if len(durations) > 1 else 0.0

        # Collect all tags across runs
        tag_data: dict[str, TagStats] = {}

        for run in successful_runs:
            seen_in_run = set()
            for tag in run.tags:
                key = f"{tag['plugin']}:{tag['label']}"
                if key not in tag_data:
                    tag_data[key] = TagStats(
                        label=key,
                        occurrence_count=0,
                        total_runs=num_successful,
                    )

                if key not in seen_in_run:
                    tag_data[key].occurrence_count += 1
                    seen_in_run.add(key)

                if tag.get("confidence") is not None:
                    tag_data[key].confidences.append(tag["confidence"])

        # Compute per-tag statistics
        for ts in tag_data.values():
            ts.compute_stats()

        tag_stats = list(tag_data.values())

        # Aggregate metrics
        total_unique = len(tag_stats)
        in_all = sum(1 for ts in tag_stats if ts.occurrence_rate == 1.0)
        in_most = sum(1 for ts in tag_stats if ts.occurrence_rate > 0.8)
        in_some = sum(1 for ts in tag_stats if ts.occurrence_rate > 0.5)

        # High variance tags (std > 0.1)
        high_variance = [
            ts.label for ts in tag_stats
            if ts.confidence_std and ts.confidence_std > 0.1
        ]

        # Consistency score: weighted by occurrence rate
        if tag_stats:
            consistency_score = sum(ts.occurrence_rate for ts in tag_stats) / len(tag_stats)
        else:
            consistency_score = 1.0

        # Check if fully deterministic
        is_deterministic = (
            in_all == total_unique and
            len(high_variance) == 0 and
            num_successful == len(self.runs)
        )

        return ComparisonReport(
            image_path=str(self.image_path),
            image_hash=self.runs[0].image_hash if self.runs else "",
            num_runs=len(self.runs),
            num_successful=num_successful,
            cli_args=self.cli_args,
            duration_mean_ms=duration_mean,
            duration_std_ms=duration_std,
            total_unique_tags=total_unique,
            tags_in_all_runs=in_all,
            tags_in_most_runs=in_most,
            tags_in_some_runs=in_some,
            consistency_score=consistency_score,
            tag_stats=tag_stats,
            runs=self.runs,
            is_deterministic=is_deterministic,
            has_high_variance_tags=high_variance,
        )

    def save_report(self, report: ComparisonReport, output_path: str | Path):
        """Save report to JSON file."""
        Path(output_path).write_text(
            json.dumps(report.to_dict(), indent=2, default=str)
        )


def compare_two_reports(report1: ComparisonReport, report2: ComparisonReport) -> dict:
    """
    Compare two reports to detect drift between versions/configs.

    Returns dict with:
        - tags_added: Tags in report2 not in report1
        - tags_removed: Tags in report1 not in report2
        - confidence_changes: Tags with significant confidence shift
    """
    tags1 = {ts.label: ts for ts in report1.tag_stats}
    tags2 = {ts.label: ts for ts in report2.tag_stats}

    labels1 = set(tags1.keys())
    labels2 = set(tags2.keys())

    added = labels2 - labels1
    removed = labels1 - labels2
    common = labels1 & labels2

    confidence_changes = []
    for label in common:
        ts1, ts2 = tags1[label], tags2[label]
        if ts1.confidence_mean and ts2.confidence_mean:
            diff = abs(ts1.confidence_mean - ts2.confidence_mean)
            if diff > 0.1:  # Significant change threshold
                confidence_changes.append({
                    "label": label,
                    "old_confidence": ts1.confidence_mean,
                    "new_confidence": ts2.confidence_mean,
                    "change": ts2.confidence_mean - ts1.confidence_mean,
                })

    return {
        "tags_added": list(added),
        "tags_removed": list(removed),
        "confidence_changes": confidence_changes,
        "jaccard_similarity": len(common) / len(labels1 | labels2) if (labels1 | labels2) else 1.0,
    }


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare multiple CLI runs")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--plugin", action="append", dest="plugins", help="Plugin(s) to use")
    parser.add_argument("--threshold", type=float, help="Confidence threshold")
    parser.add_argument("--output", help="Output JSON file for report")
    parser.add_argument("--cli", default="visual-buffet", help="CLI command")

    args = parser.parse_args()

    comparator = RunComparator(cli_command=args.cli)

    print(f"Executing {args.runs} runs on {args.image}...")
    comparator.execute_runs(
        args.image,
        num_runs=args.runs,
        plugins=args.plugins,
        threshold=args.threshold,
    )

    report = comparator.compare()
    print(report.summary())

    if args.output:
        comparator.save_report(report, args.output)
        print(f"\nFull report saved to: {args.output}")
