"""IMLAGE CLI entry point."""

import click
from rich.console import Console

from imlage import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="imlage")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: click.Context, debug: bool) -> None:
    """IMLAGE - Compare visual tagging results from local ML tools."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--plugin", "-p", multiple=True, help="Plugins to use (default: all)")
@click.option("--output", "-o", type=click.Path(), help="Output file (default: stdout)")
@click.pass_context
def tag(ctx: click.Context, path: str, plugin: tuple, output: str | None) -> None:
    """Tag image(s) using configured plugins."""
    console.print(f"[dim]Tagging: {path}[/dim]")
    console.print("[yellow]Not yet implemented[/yellow]")


@main.command()
@click.pass_context
def plugins(ctx: click.Context) -> None:
    """List available plugins."""
    console.print("[yellow]Not yet implemented[/yellow]")


@main.command()
@click.pass_context
def hardware(ctx: click.Context) -> None:
    """Show detected hardware."""
    console.print("[yellow]Not yet implemented[/yellow]")


if __name__ == "__main__":
    main()
