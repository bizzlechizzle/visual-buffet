"""Visual Buffet CLI entry point.

Commands:
    tag       Tag image(s) using configured plugins
    plugins   Manage plugins (list, setup)
    hardware  Show detected hardware
    config    View/edit configuration
"""

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from visual_buffet import __version__
from visual_buffet.constants import (
    DEFAULT_GUI_HOST,
    DEFAULT_GUI_PORT,
    DEFAULT_THRESHOLD,
    MAX_THRESHOLD,
    MIN_THRESHOLD,
    ExitCode,
)
from visual_buffet.core.engine import TaggingEngine
from visual_buffet.core.hardware import detect_hardware, get_recommended_batch_size
from visual_buffet.exceptions import VisualBuffetError
from visual_buffet.plugins.loader import discover_plugins, get_plugins_dir, load_plugin
from visual_buffet.utils.config import get_value, load_config, save_config, set_value
from visual_buffet.utils.image import expand_paths
from visual_buffet.utils.progress import ProgressState, format_duration, truncate_middle

console = Console()


def _validate_threshold(
    ctx: click.Context, param: click.Parameter, value: float
) -> float:
    """Validate threshold is within valid range."""
    if not MIN_THRESHOLD <= value <= MAX_THRESHOLD:
        raise click.BadParameter(
            f"Threshold must be between {MIN_THRESHOLD} and {MAX_THRESHOLD}"
        )
    return value


def _validate_port(ctx: click.Context, param: click.Parameter, value: int) -> int:
    """Validate port is within valid range."""
    if not 1 <= value <= 65535:
        raise click.BadParameter("Port must be between 1 and 65535")
    return value


@click.group()
@click.version_option(version=__version__, prog_name="visual-buffet")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: click.Context, debug: bool) -> None:
    """Visual Buffet - Compare visual tagging results from local ML tools."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


# ============================================================================
# TAG COMMAND
# ============================================================================


@main.command()
@click.argument("path", nargs=-1, required=True)
@click.option("-p", "--plugin", "plugins", multiple=True, help="Plugins to use")
@click.option("-o", "--output", type=click.Path(), help="Output file")
@click.option(
    "-f", "--format", "fmt", default="json", type=click.Choice(["json"]), help="Output format"
)
@click.option(
    "--threshold",
    default=DEFAULT_THRESHOLD,
    type=float,
    callback=_validate_threshold,
    help=f"Minimum confidence ({MIN_THRESHOLD}-{MAX_THRESHOLD})",
)
@click.option("--recursive", is_flag=True, help="Search folders recursively")
@click.option(
    "--size",
    type=click.Choice(["little", "small", "large", "huge", "original"]),
    default="original",
    help="Image size for tagging (default: original)",
)
@click.option(
    "--discover",
    is_flag=True,
    help="Discovery mode: SigLIP uses RAM++/Florence-2 to discover vocabulary",
)
@click.option(
    "--xmp/--no-xmp",
    default=True,
    help="Write tags to XMP sidecar (default: on)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview what would be tagged without writing files",
)
@click.option(
    "--vocab",
    type=click.Path(),
    help="Record tags to vocabulary database for learning",
)
@click.option(
    "--vocab-db",
    type=click.Path(exists=True),
    help="Use vocabulary from database for SigLIP (enables discovery mode)",
)
@click.pass_context
def tag(
    ctx: click.Context,
    path: tuple,
    plugins: tuple,
    output: str | None,
    fmt: str,
    threshold: float,
    recursive: bool,
    size: str,
    discover: bool,
    xmp: bool,
    dry_run: bool,
    vocab: str | None,
    vocab_db: str | None,
) -> None:
    """Tag image(s) using configured plugins.

    PATH can be a file, folder, or glob pattern. Multiple paths allowed.

    Examples:

        visual-buffet tag photo.jpg

        visual-buffet tag ./photos --recursive

        visual-buffet tag *.jpg -o results.json

        visual-buffet tag photo.jpg --size small

        visual-buffet tag photo.jpg --discover

        visual-buffet tag photo.jpg --xmp --vocab vocab.db
    """
    try:
        # Import XMP handler and vocab integration
        from visual_buffet.services.xmp_handler import XMPHandler

        xmp_handler = XMPHandler() if xmp else None
        vocab_integration = None

        if vocab:
            try:
                from visual_buffet.vocab_integration import VocabIntegration
                vocab_integration = VocabIntegration(vocab)
            except ImportError:
                console.print("[yellow]Warning: vocablearn not available[/yellow]")

        # Expand paths to file list
        image_paths = expand_paths(list(path), recursive=recursive)

        if not image_paths:
            console.print("[red]No images found[/red]")
            sys.exit(ExitCode.FILE_NOT_FOUND)

        # Dry run mode
        if dry_run:
            console.print("[bold yellow]DRY RUN[/bold yellow] - No files will be written")
            console.print(f"Would process {len(image_paths)} image(s)")
            console.print(f"[dim]Plugins: {', '.join(plugins) if plugins else 'all'}[/dim]")
            console.print(f"[dim]XMP output: {'yes' if xmp else 'no'}[/dim]")
            console.print(f"[dim]Vocab learning: {vocab if vocab else 'no'}[/dim]")
            for img_path in image_paths[:10]:
                console.print(f"  - {img_path}")
            if len(image_paths) > 10:
                console.print(f"  ... and {len(image_paths) - 10} more")
            return

        console.print(f"[dim]Found {len(image_paths)} image(s)[/dim]")

        # Initialize engine
        engine = TaggingEngine()

        if not engine.plugins:
            console.print("[red]No plugins available[/red]")
            console.print("Run 'visual-buffet plugins list' to see available plugins")
            console.print("Run 'visual-buffet plugins setup <name>' to set up a plugin")
            sys.exit(ExitCode.NO_PLUGINS)

        # Check if any plugins are available
        available = [name for name, p in engine.plugins.items() if p.is_available()]
        if not available:
            console.print("[yellow]No plugins are ready to use[/yellow]")
            for name, p in engine.plugins.items():
                console.print(f"  - {name}: Run 'visual-buffet plugins setup {name}'")
            sys.exit(ExitCode.NO_PLUGINS)

        # Filter to requested plugins
        plugin_names = list(plugins) if plugins else None

        # Build plugin configs
        plugin_configs = {}

        # Auto-enable discovery mode when multi-model with SigLIP
        # Discovery is enabled when: SigLIP + (RAM++ or Florence-2) are available
        if not discover and not vocab_db:
            plugins_to_check = plugin_names if plugin_names else available
            has_siglip = "siglip" in plugins_to_check
            has_discovery_source = any(
                p in plugins_to_check for p in ["ram_plus", "florence_2"]
            )
            if has_siglip and has_discovery_source:
                discover = True
                console.print(
                    "[dim]Auto-enabling discovery mode (SigLIP + discovery sources)[/dim]"
                )

        # Handle vocabulary from database (vocab-db)
        vocab_labels = None
        if vocab_db:
            try:
                from vocablearn import VocabLearn
                vl = VocabLearn(vocab_db)
                vocab_labels = vl.get_vocabulary_labels(min_occurrences=1)
                console.print(f"[bold]Vocabulary mode:[/bold] {len(vocab_labels)} tags from {vocab_db}")
                discover = True  # Auto-enable discovery with vocab-db
            except ImportError:
                console.print("[yellow]Warning: vocablearn not available[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load vocabulary: {e}[/yellow]")

        # Handle discovery mode
        if discover:
            if vocab_labels:
                console.print("[bold]Discovery mode:[/bold] SigLIP with vocabulary database")
            else:
                console.print("[bold]Discovery mode:[/bold] SigLIP with RAM++/Florence-2 vocabulary")
            console.print(f"[dim]Threshold: {threshold}[/dim]")
            console.print()

            # Ensure siglip is included
            if plugin_names and "siglip" not in plugin_names:
                plugin_names = list(plugin_names) + ["siglip"]
            elif not plugin_names:
                plugin_names = ["siglip"]

            # Configure SigLIP for discovery
            plugin_configs["siglip"] = {
                "discovery_mode": True,
                "use_ram_plus": not vocab_labels,  # Use RAM++ if no vocab-db
                "use_florence_2": not vocab_labels,  # Use Florence-2 if no vocab-db
                "custom_vocabulary": vocab_labels,  # Use vocab-db labels if available
            }
        else:
            console.print(f"[dim]Using plugins: {', '.join(available)}[/dim]")
            console.print(f"[dim]Size: {size} | Threshold: {threshold}[/dim]")
            console.print()

        # Process images with progress bar
        results = _tag_with_progress(
            engine,
            image_paths,
            plugin_names=plugin_names,
            threshold=threshold,
            plugin_configs=plugin_configs if plugin_configs else None,
            size=size,
        )

        # Post-processing: XMP and vocabulary recording
        xmp_count = 0
        vocab_count = 0
        plugins_used = list(plugins) if plugins else list(available)

        for result in results:
            if "error" in result:
                continue

            file_path = Path(result.get("file", ""))
            if not file_path.exists():
                continue

            # Collect all tags from all plugins with deduplication
            # Track unique labels with highest confidence and source plugins
            tag_map: dict[str, dict] = {}  # normalized_label -> {label, confidence, plugins}
            total_inference_ms = 0.0

            for plugin_name, plugin_result in result.get("results", {}).items():
                if "error" in plugin_result:
                    continue

                inference_ms = plugin_result.get("inference_time_ms", 0.0)
                total_inference_ms += inference_ms

                for tag in plugin_result.get("tags", []):
                    label = tag.get("label", "").strip()
                    if not label:
                        continue

                    normalized = label.lower()
                    confidence = tag.get("confidence")

                    if normalized not in tag_map:
                        tag_map[normalized] = {
                            "label": label,
                            "confidence": confidence,
                            "plugins": [plugin_name],
                        }
                    else:
                        # Track which plugins found this tag
                        if plugin_name not in tag_map[normalized]["plugins"]:
                            tag_map[normalized]["plugins"].append(plugin_name)
                        # Keep highest confidence
                        existing_conf = tag_map[normalized]["confidence"]
                        if confidence is not None:
                            if existing_conf is None or confidence > existing_conf:
                                tag_map[normalized]["confidence"] = confidence

            # Convert to list for XMP writer
            all_tags = [
                {
                    "label": v["label"],
                    "confidence": v["confidence"],
                    "plugin": ",".join(v["plugins"]),  # Comma-separated source plugins
                    "agreement": len(v["plugins"]),  # How many models agreed
                }
                for v in tag_map.values()
            ]

            # Write XMP sidecar
            if xmp_handler and all_tags:
                try:
                    if xmp_handler.write_tags(
                        image_path=file_path,
                        tags=all_tags,
                        plugins_used=plugins_used,
                        threshold=threshold,
                        size_used=size,
                        inference_time_ms=total_inference_ms,
                    ):
                        xmp_count += 1
                except Exception as e:
                    if ctx.obj.get("debug"):
                        console.print(f"[dim]XMP error for {file_path.name}: {e}[/dim]")

            # Record to vocabulary
            if vocab_integration:
                try:
                    recorded = vocab_integration.record_tagging_result(file_path, result)
                    if recorded > 0:
                        vocab_count += 1
                except Exception as e:
                    if ctx.obj.get("debug"):
                        console.print(f"[dim]Vocab error for {file_path.name}: {e}[/dim]")

        # Show post-processing summary
        if xmp_count > 0:
            console.print(f"[dim]Wrote XMP sidecars for {xmp_count} image(s)[/dim]")
        if vocab_count > 0:
            console.print(f"[dim]Recorded {vocab_count} image(s) to vocabulary[/dim]")

        # Output results
        output_json = json.dumps(results, indent=2)

        if output:
            Path(output).write_text(output_json)
            console.print(f"[green]Results saved to {output}[/green]")
        else:
            # Pretty print to console
            for result in results:
                _print_result(result)

    except VisualBuffetError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(ExitCode.GENERAL_ERROR)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(ExitCode.KEYBOARD_INTERRUPT)


class ETAColumn(TextColumn):
    """Custom column showing EWMA-smoothed ETA."""

    def __init__(self, progress_state: ProgressState):
        super().__init__("")
        self.progress_state = progress_state

    def render(self, task) -> str:
        eta_ms = self.progress_state.eta_ms
        if eta_ms is None:
            return "[dim]ETA: --[/dim]"
        if eta_ms <= 0:
            return "[dim]ETA: < 1s[/dim]"
        return f"[dim]ETA: {format_duration(eta_ms)}[/dim]"


def _tag_with_progress(
    engine: TaggingEngine,
    image_paths: list[Path],
    plugin_names: list[str] | None = None,
    threshold: float = 0.0,
    plugin_configs: dict[str, Any] | None = None,
    size: str = "original",
) -> list[dict[str, Any]]:
    """Tag images with progress bar display.

    Uses EWMA-smoothed ETA for stable time remaining estimates.
    Shows progress bar for batches > 1 image, spinner for single image.
    """
    total = len(image_paths)

    # Single image: use simple spinner (no progress bar for < 4s tasks)
    if total == 1:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(f"Tagging {image_paths[0].name}...", total=None)
            return engine.tag_batch(
                image_paths,
                plugin_names=plugin_names,
                threshold=threshold,
                plugin_configs=plugin_configs,
                size=size,
            )

    # Multiple images: show progress bar with ETA
    progress_state = ProgressState(files_total=total)
    eta_column = ETAColumn(progress_state)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        eta_column,
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Tagging", total=total)

        def on_progress(path: Path, result: dict[str, Any]) -> None:
            """Callback to update progress after each file."""
            # Calculate total inference time from all plugins
            total_time_ms = 0.0
            if "results" in result:
                for plugin_result in result["results"].values():
                    if isinstance(plugin_result, dict) and "inference_time_ms" in plugin_result:
                        total_time_ms += plugin_result["inference_time_ms"]

            # Update EWMA state
            progress_state.update(path.name, total_time_ms or 100.0)

            # Update rich progress bar
            progress.update(
                task,
                advance=1,
                description=f"[bold blue]{truncate_middle(path.name, 25)}",
            )

        results = engine.tag_batch(
            image_paths,
            plugin_names=plugin_names,
            threshold=threshold,
            plugin_configs=plugin_configs,
            size=size,
            on_progress=on_progress,
        )

        # Final update to show completion
        progress.update(task, description="[bold green]Complete")

    # Print summary
    elapsed = format_duration(progress_state.elapsed_ms)
    avg_time = progress_state.avg_time_per_file_ms
    console.print(
        f"[dim]Processed {total} images in {elapsed} "
        f"(avg {avg_time:.0f}ms/image)[/dim]"
    )

    return results


def _print_result(result: dict) -> None:
    """Pretty print a tagging result."""
    file_path = result.get("file", "unknown")
    console.print(f"\n[bold]{file_path}[/bold]")

    if "error" in result:
        console.print(f"  [red]Error: {result['error']}[/red]")
        return

    for plugin_name, plugin_result in result.get("results", {}).items():
        if "error" in plugin_result:
            console.print(f"  [cyan]{plugin_name}:[/cyan] [red]{plugin_result['error']}[/red]")
            continue

        tags = plugin_result.get("tags", [])
        metadata = plugin_result.get("metadata", {})

        # Check if this is a discovery mode result
        if metadata.get("discovery_mode"):
            _print_discovery_result(plugin_name, plugin_result)
            continue

        if not tags:
            console.print(f"  [cyan]{plugin_name}:[/cyan] [dim]No tags above threshold[/dim]")
            continue

        # Show inference time
        time_ms = plugin_result.get("inference_time_ms", 0)
        console.print(f"  [cyan]{plugin_name}[/cyan] [dim]({time_ms:.0f}ms)[/dim]:")

        # Format tags nicely (handle None confidence)
        def fmt_tag(t):
            if t.get("confidence") is not None:
                return f"{t['label']} ({t['confidence']:.2f})"
            return t["label"]

        # Show ALL tags
        all_tag_strs = [fmt_tag(t) for t in tags]
        tag_str = " • ".join(all_tag_strs)
        console.print(f"    {tag_str}")


def _print_discovery_result(plugin_name: str, plugin_result: dict) -> None:
    """Pretty print a SigLIP discovery mode result with all model outputs."""
    metadata = plugin_result.get("metadata", {})
    discovery_results = metadata.get("discovery_results", {})
    sources = metadata.get("discovery_sources", [])
    vocab_size = metadata.get("vocabulary_size", 0)
    discovery_time = metadata.get("total_discovery_time_ms", 0)

    console.print(f"  [dim]Discovery: {vocab_size} candidates from {', '.join(sources)} ({discovery_time:.0f}ms)[/dim]")

    # Print discovery plugin results (RAM++, Florence-2)
    for source_name in ["ram_plus", "florence_2"]:
        if source_name not in discovery_results:
            continue

        source_result = discovery_results[source_name]
        tags = source_result.get("tags", [])
        time_ms = source_result.get("inference_time_ms", 0)
        total_tags = source_result.get("total_tags", len(tags))

        console.print(f"  [cyan]{source_name}[/cyan] [green]\\[discovery][/green] [dim]({total_tags} tags, {time_ms:.0f}ms)[/dim]:")

        def fmt_tag(t):
            conf = t.get("confidence")
            if conf is not None:
                return f"{t['label']} ({conf:.2f})"
            return t["label"]

        all_tag_strs = [fmt_tag(t) for t in tags]
        tag_str = " • ".join(all_tag_strs)
        console.print(f"    {tag_str}")

    # Print SigLIP scored results
    tags = plugin_result.get("tags", [])
    time_ms = plugin_result.get("inference_time_ms", 0)

    if not tags:
        console.print(f"  [cyan]{plugin_name}[/cyan] [magenta]\\[scorer][/magenta] [dim]No tags above threshold[/dim]")
        return

    console.print(f"  [cyan]{plugin_name}[/cyan] [magenta]\\[scorer][/magenta] [dim](scored {vocab_size} in {time_ms:.0f}ms)[/dim]:")

    def fmt_tag(t):
        conf = t.get("confidence")
        if conf is not None:
            return f"{t['label']} ({conf:.2f})"
        return t["label"]

    all_tag_strs = [fmt_tag(t) for t in tags]
    tag_str = " • ".join(all_tag_strs)
    console.print(f"    {tag_str}")


# ============================================================================
# PLUGINS COMMAND
# ============================================================================


@main.group()
def plugins() -> None:
    """Manage plugins."""
    pass


@plugins.command("list")
def plugins_list() -> None:
    """List available plugins."""
    discovered = discover_plugins()

    if not discovered:
        console.print("[yellow]No plugins found in plugins/ directory[/yellow]")
        console.print(f"[dim]Looking in: {get_plugins_dir()}[/dim]")
        return

    table = Table(title="Available Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Status")
    table.add_column("Description")

    for meta in discovered:
        plugin_section = meta.get("plugin", {})
        name = plugin_section.get("name", "unknown")
        version = plugin_section.get("version", "?")
        description = plugin_section.get("description", "")[:50]

        # Check if available
        try:
            plugin = load_plugin(meta["_path"])
            if plugin.is_available():
                status = "[green]Ready[/green]"
            else:
                status = "[yellow]Setup needed[/yellow]"
        except Exception:
            status = "[red]Error[/red]"

        table.add_row(name, version, status, description)

    console.print(table)


@plugins.command("setup")
@click.argument("name")
def plugins_setup(name: str) -> None:
    """Download and setup a plugin's model files.

    NAME is the plugin name (e.g., ram_plus)
    """
    plugins_dir = get_plugins_dir()
    plugin_dir = plugins_dir / name

    if not plugin_dir.exists():
        console.print(f"[red]Plugin '{name}' not found[/red]")
        console.print(f"[dim]Looking in: {plugins_dir}[/dim]")
        console.print("\nAvailable plugins:")
        for meta in discover_plugins():
            plugin_name = meta.get("plugin", {}).get("name", "unknown")
            console.print(f"  - {plugin_name}")
        sys.exit(1)

    try:
        plugin = load_plugin(plugin_dir)

        if plugin.is_available():
            console.print(f"[green]{name} is already set up![/green]")
            return

        console.print(f"Setting up {name}...")
        success = plugin.setup()

        if success:
            console.print(f"[green]{name} setup complete![/green]")
        else:
            console.print(f"[red]{name} setup failed[/red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Setup error: {e}[/red]")
        if ctx := click.get_current_context():
            if ctx.obj and ctx.obj.get("debug"):
                import traceback

                console.print(traceback.format_exc())
        sys.exit(1)


@plugins.command("info")
@click.argument("name")
def plugins_info(name: str) -> None:
    """Show detailed information about a plugin.

    NAME is the plugin name (e.g., ram_plus)
    """
    plugins_dir = get_plugins_dir()
    plugin_dir = plugins_dir / name

    if not plugin_dir.exists():
        console.print(f"[red]Plugin '{name}' not found[/red]")
        sys.exit(1)

    try:
        plugin = load_plugin(plugin_dir)
        info = plugin.get_info()

        console.print(f"\n[bold]{info.name}[/bold] v{info.version}")
        console.print(f"[dim]{info.description}[/dim]")
        console.print()

        # Status
        if plugin.is_available():
            console.print("[green]✓ Ready to use[/green]")
        else:
            console.print("[yellow]⚠ Setup required[/yellow]")
            console.print(f"  Run: visual-buffet plugins setup {name}")

        # Hardware requirements
        if info.hardware_reqs:
            console.print("\n[bold]Hardware Requirements:[/bold]")
            for key, value in info.hardware_reqs.items():
                console.print(f"  {key}: {value}")

        # Model path
        model_path = plugin.get_model_path()
        console.print(f"\n[bold]Model Path:[/bold] {model_path}")

        # List model files
        if model_path.exists():
            files = list(model_path.iterdir())
            if files:
                console.print("\n[bold]Model Files:[/bold]")
                for f in files:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    console.print(f"  {f.name} ({size_mb:.1f} MB)")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# ============================================================================
# HARDWARE COMMAND
# ============================================================================


@main.command()
@click.option("--refresh", is_flag=True, help="Force re-detection")
def hardware(refresh: bool) -> None:
    """Show detected hardware capabilities."""
    try:
        console.print("[dim]Detecting hardware...[/dim]")
        profile = detect_hardware(force_refresh=refresh)

        table = Table(title="Hardware Profile")
        table.add_column("Component", style="cyan")
        table.add_column("Details")

        table.add_row("CPU", f"{profile.cpu_model}")
        table.add_row("CPU Cores", f"{profile.cpu_cores}")
        table.add_row(
            "RAM", f"{profile.ram_total_gb} GB total, {profile.ram_available_gb} GB available"
        )

        if profile.gpu_type:
            gpu_info = f"{profile.gpu_name}"
            if profile.gpu_vram_gb:
                gpu_info += f" ({profile.gpu_vram_gb} GB VRAM)"
            table.add_row(f"GPU ({profile.gpu_type.upper()})", gpu_info)
        else:
            table.add_row("GPU", "[dim]None detected[/dim]")

        console.print(table)

        # Recommendations
        batch_size = get_recommended_batch_size(profile)
        console.print("\n[bold]Recommendations:[/bold]")
        console.print(f"  Batch size: {batch_size}")

        if profile.gpu_type:
            console.print("  [green]GPU acceleration available[/green]")
        else:
            console.print("  [yellow]CPU-only mode (slower)[/yellow]")

    except Exception as e:
        console.print(f"[red]Hardware detection failed: {e}[/red]")
        sys.exit(1)


# ============================================================================
# CONFIG COMMAND
# ============================================================================


@main.group()
def config() -> None:
    """View and edit configuration."""
    pass


@config.command("show")
def config_show() -> None:
    """Show current configuration."""
    from visual_buffet.utils.config import get_config_path

    cfg = load_config()
    console.print(f"[dim]Config file: {get_config_path()}[/dim]\n")
    console.print_json(json.dumps(cfg, indent=2))


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a configuration value.

    KEY is a dot-separated path (e.g., general.default_threshold)
    VALUE is the new value
    """
    cfg = load_config()

    # Try to parse as JSON for complex values
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        # Try to convert to number if possible
        try:
            parsed_value = float(value)
            if parsed_value.is_integer():
                parsed_value = int(parsed_value)
        except ValueError:
            parsed_value = value

    set_value(cfg, key, parsed_value)
    save_config(cfg)
    console.print(f"[green]Set {key} = {parsed_value}[/green]")


@config.command("get")
@click.argument("key")
def config_get(key: str) -> None:
    """Get a configuration value.

    KEY is a dot-separated path (e.g., general.default_threshold)
    """
    cfg = load_config()
    value = get_value(cfg, key)

    if value is None:
        console.print(f"[yellow]Key '{key}' not found[/yellow]")
        sys.exit(1)

    console.print(f"{key} = {json.dumps(value)}")


# ============================================================================
# GUI COMMAND
# ============================================================================


@main.command()
@click.option("--host", default=DEFAULT_GUI_HOST, help="Host to bind to")
@click.option(
    "--port",
    default=DEFAULT_GUI_PORT,
    type=int,
    callback=_validate_port,
    help="Port to bind to",
)
@click.option("--no-browser", is_flag=True, help="Don't open browser automatically")
def gui(host: str, port: int, no_browser: bool) -> None:
    """Launch the web GUI.

    Opens a browser window with the Visual Buffet interface.

    Examples:

        visual-buffet gui

        visual-buffet gui --port 9000

        visual-buffet gui --no-browser
    """
    try:
        import uvicorn
    except ImportError:
        console.print("[red]GUI dependencies not installed[/red]")
        console.print("Install with: pip install 'visual-buffet[gui]'")
        console.print("  or: pip install fastapi uvicorn")
        sys.exit(1)

    from visual_buffet.gui import app

    url = f"http://{host}:{port}"
    console.print("\n[bold]Visual Buffet GUI[/bold]")
    console.print(f"Starting server at [cyan]{url}[/cyan]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    # Open browser
    if not no_browser:
        import threading
        import time
        import webbrowser

        def open_browser():
            time.sleep(1)  # Wait for server to start
            webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning",
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")


# ============================================================================
# VOCAB COMMAND (Vocabulary Learning)
# ============================================================================


def _resolve_vocab_db(app: str | None, db_path: str) -> str:
    """Resolve vocabulary database path from --app or --db."""
    if app:
        from vocablearn import AppConfig
        config = AppConfig(app)
        return str(config.vocab_db)
    return db_path


def _resolve_ocr_db(app: str | None, db_path: str) -> str:
    """Resolve OCR database path from --app or --db."""
    if app:
        from vocablearn import AppConfig
        config = AppConfig(app)
        return str(config.ocr_db)
    return db_path


@main.group()
def vocab() -> None:
    """Vocabulary learning commands.

    Track and learn from tagging results over time.

    Use --app to specify an app name for isolated database storage:

        visual-buffet vocab stats --app abandoned-archive

    This stores data in ~/.abandoned-archive/data/vocabulary.db
    """
    pass


@vocab.command("stats")
@click.option("--app", help="App name for isolated database (e.g., abandoned-archive)")
@click.option("--db", "db_path", default="vocabulary.db", help="Vocabulary database path")
def vocab_stats(app: str | None, db_path: str) -> None:
    """Show vocabulary statistics.

    Displays counts of tags, feedback, calibration data, and model agreement.
    """
    try:
        from visual_buffet.vocab_integration import VocabIntegration

        resolved_db = _resolve_vocab_db(app, db_path)
        vocab = VocabIntegration(resolved_db)
        stats = vocab.get_statistics()

        table = Table(title="Vocabulary Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total Tags", str(stats.get("total_tags", 0)))
        table.add_row("Total Events", str(stats.get("total_events", 0)))
        table.add_row("Unique Images", str(stats.get("unique_images", 0)))
        table.add_row("Feedback Count", str(stats.get("feedback_count", 0)))
        table.add_row("Calibration Points", str(stats.get("calibration_points", 0)))

        if "model_counts" in stats:
            table.add_row("", "")  # Separator
            for model, count in stats["model_counts"].items():
                table.add_row(f"  {model}", str(count))

        console.print(table)

    except ImportError:
        console.print("[red]vocablearn not installed[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@vocab.command("search")
@click.argument("query", required=False)
@click.option("--app", help="App name for isolated database (e.g., abandoned-archive)")
@click.option("--db", "db_path", default="vocabulary.db", help="Vocabulary database path")
@click.option("--min-count", default=1, help="Minimum occurrence count")
@click.option("--limit", default=50, help="Maximum results")
def vocab_search(query: str | None, app: str | None, db_path: str, min_count: int, limit: int) -> None:
    """Search vocabulary entries.

    QUERY is an optional prefix to search for.
    """
    try:
        from visual_buffet.vocab_integration import VocabIntegration

        resolved_db = _resolve_vocab_db(app, db_path)
        vocab = VocabIntegration(resolved_db)
        tags = vocab.vocab.search_vocabulary(
            query=query,
            min_occurrences=min_count,
            limit=limit,
        )

        if not tags:
            console.print("[dim]No tags found[/dim]")
            return

        table = Table(title=f"Vocabulary Search{f': {query}' if query else ''}")
        table.add_column("Tag", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Prior", justify="right")
        table.add_column("RAM++", justify="right")
        table.add_column("Florence", justify="right")
        table.add_column("SigLIP", justify="right")

        for tag in tags:
            table.add_row(
                tag.label,
                str(tag.total_occurrences),
                f"{tag.prior_confidence:.2f}",
                str(tag.ram_plus_count),
                str(tag.florence_2_count),
                str(tag.siglip_count),
            )

        console.print(table)

    except ImportError:
        console.print("[red]vocablearn not installed[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@vocab.command("export")
@click.argument("output", type=click.Path())
@click.option("--app", help="App name for isolated database (e.g., abandoned-archive)")
@click.option("--db", "db_path", default="vocabulary.db", help="Vocabulary database path")
def vocab_export(output: str, app: str | None, db_path: str) -> None:
    """Export vocabulary to JSON file.

    OUTPUT is the destination file path.
    """
    try:
        from visual_buffet.vocab_integration import VocabIntegration

        resolved_db = _resolve_vocab_db(app, db_path)
        vocab = VocabIntegration(resolved_db)
        vocab.export_vocabulary(output)
        console.print(f"[green]Vocabulary exported to {output}[/green]")

    except ImportError:
        console.print("[red]vocablearn not installed[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@vocab.command("import")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--app", help="App name for isolated database (e.g., abandoned-archive)")
@click.option("--db", "db_path", default="vocabulary.db", help="Vocabulary database path")
@click.option("--merge/--replace", default=True, help="Merge with existing or replace")
def vocab_import(input_file: str, app: str | None, db_path: str, merge: bool) -> None:
    """Import vocabulary from JSON file.

    INPUT_FILE is the source vocabulary file.
    """
    try:
        from visual_buffet.vocab_integration import VocabIntegration

        resolved_db = _resolve_vocab_db(app, db_path)
        vocab = VocabIntegration(resolved_db)
        count = vocab.import_vocabulary(input_file, merge=merge)
        console.print(f"[green]Imported {count} tags[/green]")

    except ImportError:
        console.print("[red]vocablearn not installed[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@vocab.command("learn")
@click.option("--app", help="App name for isolated database (e.g., abandoned-archive)")
@click.option("--db", "db_path", default="vocabulary.db", help="Vocabulary database path")
@click.option("--min-samples", default=5, help="Minimum samples for prior updates")
def vocab_learn(app: str | None, db_path: str, min_samples: int) -> None:
    """Update priors and calibrators from feedback.

    Recalculates Bayesian priors and rebuilds isotonic regression
    calibrators from accumulated human feedback.
    """
    try:
        from visual_buffet.vocab_integration import VocabIntegration

        resolved_db = _resolve_vocab_db(app, db_path)
        vocab = VocabIntegration(resolved_db)
        result = vocab.update_learning(min_samples=min_samples)

        console.print(f"[green]Updated {result['priors_updated']} priors[/green]")
        console.print(f"[green]Rebuilt {result['calibrators_rebuilt']} calibrators[/green]")

    except ImportError:
        console.print("[red]vocablearn not installed[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@vocab.command("review")
@click.option("--app", help="App name for isolated database (e.g., abandoned-archive)")
@click.option("--db", "db_path", default="vocabulary.db", help="Vocabulary database path")
@click.option("-n", "--count", default=10, help="Number of images to select")
@click.option(
    "--strategy",
    type=click.Choice(["uncertainty", "diversity", "high_volume"]),
    default="uncertainty",
    help="Selection strategy",
)
def vocab_review(app: str | None, db_path: str, count: int, strategy: str) -> None:
    """Select images for human review.

    Uses active learning to prioritize images that would
    provide the most learning value.

    Strategies:
      uncertainty: Images with uncertain predictions
      diversity: Images representing diverse tag distributions
      high_volume: High-frequency tags needing validation
    """
    try:
        from visual_buffet.vocab_integration import VocabIntegration

        resolved_db = _resolve_vocab_db(app, db_path)
        vocab = VocabIntegration(resolved_db)
        candidates = vocab.select_for_review(n=count, strategy=strategy)

        if not candidates:
            console.print("[dim]No candidates found for review[/dim]")
            return

        table = Table(title=f"Review Candidates ({strategy})")
        table.add_column("Image ID", style="cyan")
        table.add_column("Priority", justify="right")
        table.add_column("Uncertain Tags")

        for candidate in candidates:
            tags = ", ".join(candidate.get("uncertain_tags", [])[:5])
            if len(candidate.get("uncertain_tags", [])) > 5:
                tags += "..."
            table.add_row(
                candidate.get("image_id", "")[:16],
                f"{candidate.get('priority', 0):.2f}",
                tags,
            )

        console.print(table)

    except ImportError:
        console.print("[red]vocablearn not installed[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
