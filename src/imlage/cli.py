"""IMLAGE CLI entry point.

Commands:
    tag       Tag image(s) using configured plugins
    plugins   Manage plugins (list, setup)
    hardware  Show detected hardware
    config    View/edit configuration
"""

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from imlage import __version__
from imlage.core.engine import TaggingEngine
from imlage.core.hardware import detect_hardware, get_recommended_batch_size
from imlage.exceptions import ImlageError
from imlage.plugins.loader import discover_plugins, get_plugins_dir, load_plugin
from imlage.utils.config import get_value, load_config, save_config, set_value
from imlage.utils.image import expand_paths

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="imlage")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: click.Context, debug: bool) -> None:
    """IMLAGE - Compare visual tagging results from local ML tools."""
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
@click.option("--threshold", default=0.5, type=float, help="Minimum confidence (0.0-1.0)")
@click.option("--limit", default=50, type=int, help="Max tags per plugin")
@click.option("--recursive", is_flag=True, help="Search folders recursively")
@click.pass_context
def tag(
    ctx: click.Context,
    path: tuple,
    plugins: tuple,
    output: str | None,
    fmt: str,
    threshold: float,
    limit: int,
    recursive: bool,
) -> None:
    """Tag image(s) using configured plugins.

    PATH can be a file, folder, or glob pattern. Multiple paths allowed.

    Examples:

        imlage tag photo.jpg

        imlage tag ./photos --recursive

        imlage tag *.jpg -o results.json
    """
    try:
        # Expand paths to file list
        image_paths = expand_paths(list(path), recursive=recursive)

        if not image_paths:
            console.print("[red]No images found[/red]")
            sys.exit(1)

        console.print(f"[dim]Found {len(image_paths)} image(s)[/dim]")

        # Initialize engine
        engine = TaggingEngine()

        if not engine.plugins:
            console.print("[red]No plugins available[/red]")
            console.print("Run 'imlage plugins list' to see available plugins")
            console.print("Run 'imlage plugins setup <name>' to set up a plugin")
            sys.exit(1)

        # Check if any plugins are available
        available = [name for name, p in engine.plugins.items() if p.is_available()]
        if not available:
            console.print("[yellow]No plugins are ready to use[/yellow]")
            for name, p in engine.plugins.items():
                console.print(f"  - {name}: Run 'imlage plugins setup {name}'")
            sys.exit(1)

        # Filter to requested plugins
        plugin_names = list(plugins) if plugins else None

        # Process images
        console.print(f"[dim]Using plugins: {', '.join(available)}[/dim]")
        console.print(f"[dim]Threshold: {threshold}, Limit: {limit}[/dim]")
        console.print()

        results = engine.tag_batch(
            image_paths,
            plugin_names=plugin_names,
            threshold=threshold,
            limit=limit,
        )

        # Output results
        output_json = json.dumps(results, indent=2)

        if output:
            Path(output).write_text(output_json)
            console.print(f"[green]Results saved to {output}[/green]")
        else:
            # Pretty print to console
            for result in results:
                _print_result(result)

    except ImlageError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(130)


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

        tag_str = " • ".join(fmt_tag(t) for t in tags[:10])
        console.print(f"    {tag_str}")

        if len(tags) > 10:
            console.print(f"    [dim]... and {len(tags) - 10} more[/dim]")


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
            console.print(f"  Run: imlage plugins setup {name}")

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
    from imlage.utils.config import get_config_path

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
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8420, type=int, help="Port to bind to")
@click.option("--no-browser", is_flag=True, help="Don't open browser automatically")
def gui(host: str, port: int, no_browser: bool) -> None:
    """Launch the web GUI.

    Opens a browser window with the IMLAGE interface.

    Examples:

        imlage gui

        imlage gui --port 9000

        imlage gui --no-browser
    """
    try:
        import uvicorn
    except ImportError:
        console.print("[red]GUI dependencies not installed[/red]")
        console.print("Install with: pip install 'imlage[gui]'")
        console.print("  or: pip install fastapi uvicorn")
        sys.exit(1)

    from imlage.gui import app

    url = f"http://{host}:{port}"
    console.print("\n[bold]IMLAGE GUI[/bold]")
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


if __name__ == "__main__":
    main()
