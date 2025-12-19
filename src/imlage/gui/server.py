"""FastAPI server for IMLAGE GUI.

Serves the web interface and handles API requests for tagging.
"""

import atexit
import io
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import rawpy
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from ..core.engine import TaggingEngine
from ..core.hardware import detect_hardware, get_recommended_batch_size
from ..plugins.loader import discover_plugins, get_plugins_dir, load_plugin
from ..utils.config import load_config
from ..utils.image import RAW_EXTENSIONS, SUPPORTED_EXTENSIONS

# App instance
app = FastAPI(
    title="IMLAGE",
    description="Image Machine Learning Aggregate",
    version="0.1.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"

# Create temp cache directory for thumbnails (wiped on each startup)
CACHE_DIR = Path(tempfile.mkdtemp(prefix="imlage_cache_"))
THUMB_DIR = CACHE_DIR / "thumbnails"
THUMB_DIR.mkdir(exist_ok=True)


def cleanup_cache():
    """Clean up the cache directory on exit."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR, ignore_errors=True)


# Register cleanup on exit
atexit.register(cleanup_cache)

# In-memory storage for uploaded images and results
_sessions: dict[str, dict[str, Any]] = {}


def get_engine() -> TaggingEngine:
    """Get or create tagging engine."""
    return TaggingEngine()


# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve main page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"error": "Frontend not found"}, status_code=404)


@app.get("/api/status")
async def status():
    """Get system status."""
    try:
        profile = detect_hardware(force_refresh=False)
        plugins = discover_plugins()

        # Check plugin availability
        plugin_status = []
        for meta in plugins:
            plugin_section = meta.get("plugin", {})
            name = plugin_section.get("name", "unknown")
            display_name = plugin_section.get("display_name", name)
            provides_confidence = plugin_section.get("provides_confidence", True)
            try:
                plugin = load_plugin(meta["_path"])
                available = plugin.is_available()
            except Exception:
                available = False

            plugin_status.append({
                "name": name,
                "display_name": display_name,
                "version": plugin_section.get("version", "?"),
                "description": plugin_section.get("description", ""),
                "available": available,
                "provides_confidence": provides_confidence,
            })

        return {
            "status": "ok",
            "hardware": {
                "cpu": profile.cpu_model,
                "cpu_cores": profile.cpu_cores,
                "ram_total_gb": profile.ram_total_gb,
                "ram_available_gb": profile.ram_available_gb,
                "gpu_type": profile.gpu_type,
                "gpu_name": profile.gpu_name,
                "gpu_vram_gb": profile.gpu_vram_gb,
            },
            "batch_size": get_recommended_batch_size(profile),
            "plugins": plugin_status,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/plugins")
async def get_plugins():
    """List available plugins."""
    plugins = discover_plugins()
    result = []

    for meta in plugins:
        plugin_section = meta.get("plugin", {})
        name = plugin_section.get("name", "unknown")
        display_name = plugin_section.get("display_name", name)

        try:
            plugin = load_plugin(meta["_path"])
            available = plugin.is_available()
            info = plugin.get_info()
            hardware_reqs = info.hardware_reqs or {}
        except Exception:
            available = False
            hardware_reqs = {}

        result.append({
            "name": name,
            "display_name": display_name,
            "version": plugin_section.get("version", "?"),
            "description": plugin_section.get("description", ""),
            "available": available,
            "hardware_reqs": hardware_reqs,
        })

    return {"plugins": result}


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    cfg = load_config()
    return cfg


def _is_raw_filename(filename: str) -> bool:
    """Check if filename has a RAW extension."""
    if not filename:
        return False
    ext = Path(filename).suffix.lower()
    return ext in RAW_EXTENSIONS


def _is_supported_filename(filename: str) -> bool:
    """Check if filename has a supported image extension."""
    if not filename:
        return False
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for tagging."""
    try:
        # Read file
        contents = await file.read()
        filename = file.filename or "unknown"
        is_raw = _is_raw_filename(filename)
        is_supported = _is_supported_filename(filename)

        # Validate file type - allow image/* MIME or any supported extension
        if not is_supported:
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(400, f"Unsupported file type: {Path(filename).suffix}")

        # Generate unique ID
        image_id = str(uuid.uuid4())[:8]

        # Validate and get image info
        if is_raw:
            # RAW file: write to temp, process with rawpy
            ext = Path(filename).suffix.lower()
            tmp_raw_path = CACHE_DIR / f"upload_{image_id}{ext}"
            tmp_raw_path.write_bytes(contents)

            try:
                with rawpy.imread(str(tmp_raw_path)) as raw:
                    # Get dimensions from raw sizes
                    width, height = raw.sizes.width, raw.sizes.height
                    # Generate thumbnail using half-size for speed
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        half_size=True,
                        output_bps=8,
                    )
                img = Image.fromarray(rgb)
                img_format = ext.upper().lstrip(".")
            except Exception as e:
                tmp_raw_path.unlink(missing_ok=True)
                raise HTTPException(400, f"Invalid RAW file: {e}")
            finally:
                tmp_raw_path.unlink(missing_ok=True)
        else:
            # Standard image: use Pillow
            try:
                img = Image.open(io.BytesIO(contents))
                img.verify()
                # Re-open after verify
                img = Image.open(io.BytesIO(contents))
                width, height = img.width, img.height
                img_format = img.format or "JPEG"
            except Exception:
                raise HTTPException(400, "Invalid image file")

        # Convert to RGB if necessary (for JPEG saving)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Create grid thumbnail (200px for grid display)
        thumb = img.copy()
        thumb.thumbnail((200, 200))
        thumb_path = THUMB_DIR / f"{image_id}.jpg"
        thumb.save(thumb_path, format="JPEG", quality=85)

        # Create preview image for lightbox (1920px max)
        preview = img.copy()
        max_preview = 1920
        if preview.width > max_preview or preview.height > max_preview:
            preview.thumbnail((max_preview, max_preview), Image.Resampling.LANCZOS)
        preview_path = CACHE_DIR / f"preview_{image_id}.jpg"
        preview.save(preview_path, format="JPEG", quality=90)

        # Get original file extension for temp file creation during tagging
        original_ext = Path(filename).suffix.lower() or ".jpg"

        # Store in session
        _sessions[image_id] = {
            "filename": filename,
            "data": contents,
            "width": width,
            "height": height,
            "format": img_format,
            "original_ext": original_ext,
            "thumb_path": thumb_path,
            "preview_path": preview_path,
            "results": None,
        }

        return {
            "id": image_id,
            "filename": filename,
            "width": width,
            "height": height,
            "format": img_format,
            "thumbnail": f"/api/thumbnail/{image_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")


@app.get("/api/thumbnail/{image_id}")
async def get_thumbnail(image_id: str):
    """Get thumbnail image file."""
    if image_id not in _sessions:
        raise HTTPException(404, "Image not found")

    session = _sessions[image_id]
    thumb_path = session.get("thumb_path")

    if not thumb_path or not thumb_path.exists():
        raise HTTPException(404, "Thumbnail not found")

    return FileResponse(
        thumb_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    """Get full image data as file (converted to displayable format for RAW)."""
    if image_id not in _sessions:
        raise HTTPException(404, "Image not found")

    session = _sessions[image_id]
    img_data = session["data"]
    original_ext = session.get("original_ext", ".jpg").lower()

    # Check if we have a cached preview
    preview_path = session.get("preview_path")
    if preview_path and preview_path.exists():
        return FileResponse(
            preview_path,
            media_type="image/jpeg",
            filename=Path(session["filename"]).stem + ".jpg",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    # For RAW files, convert to JPEG for browser display
    if original_ext in RAW_EXTENSIONS:
        # Write RAW to temp, convert with rawpy
        tmp_raw_path = CACHE_DIR / f"raw_{image_id}{original_ext}"
        tmp_raw_path.write_bytes(img_data)

        try:
            with rawpy.imread(str(tmp_raw_path)) as raw:
                # Full quality processing for display
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=False,
                    output_bps=8,
                )
            img = Image.fromarray(rgb)

            # Create preview at reasonable size (max 1920px)
            max_size = 1920
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Save as JPEG
            preview_path = CACHE_DIR / f"preview_{image_id}.jpg"
            img.save(preview_path, format="JPEG", quality=90)

            # Cache the preview path
            session["preview_path"] = preview_path

            return FileResponse(
                preview_path,
                media_type="image/jpeg",
                filename=Path(session["filename"]).stem + ".jpg",
                headers={"Cache-Control": "private, max-age=3600"},
            )
        except Exception as e:
            raise HTTPException(500, f"Failed to convert RAW file: {e}")
        finally:
            tmp_raw_path.unlink(missing_ok=True)
    else:
        # Standard image - serve directly
        img_format = session["format"].lower()
        temp_path = CACHE_DIR / f"full_{image_id}.{img_format}"
        temp_path.write_bytes(img_data)

        # Determine MIME type
        mime_types = {
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
            "bmp": "image/bmp",
            "tiff": "image/tiff",
            "tif": "image/tiff",
        }
        mime_type = mime_types.get(img_format, f"image/{img_format}")

        return FileResponse(
            temp_path,
            media_type=mime_type,
            filename=session["filename"],
            headers={"Cache-Control": "private, max-age=3600"},
        )


@app.get("/api/image/{image_id}/meta")
async def get_image_meta(image_id: str):
    """Get image metadata and results (without image data)."""
    if image_id not in _sessions:
        raise HTTPException(404, "Image not found")

    session = _sessions[image_id]

    return {
        "id": image_id,
        "filename": session["filename"],
        "width": session["width"],
        "height": session["height"],
        "format": session["format"],
        "thumbnail": f"/api/thumbnail/{image_id}",
        "results": session.get("results"),
    }


from pydantic import BaseModel
from typing import Literal


class PluginConfig(BaseModel):
    threshold: float = 0.5
    limit: int = 50
    quality: Literal["quick", "standard", "high", "max"] = "standard"


class TagRequest(BaseModel):
    plugins: list[str] | None = None
    plugin_configs: dict[str, PluginConfig] | None = None


@app.post("/api/tag/{image_id}")
async def tag_image(image_id: str, request: TagRequest | None = None):
    """Tag an uploaded image with per-plugin settings."""
    if image_id not in _sessions:
        raise HTTPException(404, "Image not found")

    session = _sessions[image_id]

    # Extract settings from request
    plugins = request.plugins if request else None
    plugin_configs = request.plugin_configs if request else None

    try:
        # Write to temp file for processing - use original extension for RAW support
        original_ext = session.get("original_ext", ".jpg")
        tmp_path = CACHE_DIR / f"tag_{image_id}{original_ext}"
        tmp_path.write_bytes(session["data"])

        try:
            # Run tagging with per-plugin configs
            # Note: save_tags=False because GUI stores results in memory,
            # and tmp_path is a temp file that gets cleaned up anyway
            engine = get_engine()
            result = engine.tag_image(
                tmp_path,
                plugin_names=plugins,
                plugin_configs=plugin_configs,
                save_tags=False,
            )

            # Update filename in result
            result["file"] = session["filename"]

            # Store results
            session["results"] = result

            return result

        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

    except Exception as e:
        raise HTTPException(500, f"Tagging failed: {str(e)}")


@app.post("/api/tag-batch")
async def tag_batch(
    image_ids: list[str],
    plugins: list[str] | None = None,
    threshold: float = 0.5,
    limit: int = 50,
):
    """Tag multiple uploaded images."""
    results = []

    for image_id in image_ids:
        if image_id not in _sessions:
            results.append({"id": image_id, "error": "Image not found"})
            continue

        try:
            result = await tag_image(image_id, plugins, threshold, limit)
            result["id"] = image_id
            results.append(result)
        except Exception as e:
            results.append({"id": image_id, "error": str(e)})

    return {"results": results}


@app.delete("/api/image/{image_id}")
async def delete_image(image_id: str):
    """Delete an uploaded image."""
    if image_id not in _sessions:
        raise HTTPException(404, "Image not found")

    session = _sessions[image_id]

    # Clean up thumbnail file
    thumb_path = session.get("thumb_path")
    if thumb_path and thumb_path.exists():
        thumb_path.unlink(missing_ok=True)

    # Clean up preview file
    preview_path = session.get("preview_path")
    if preview_path and preview_path.exists():
        preview_path.unlink(missing_ok=True)

    del _sessions[image_id]
    return {"status": "deleted"}


@app.get("/api/images")
async def list_images():
    """List all uploaded images."""
    images = []
    for image_id, session in _sessions.items():
        images.append({
            "id": image_id,
            "filename": session["filename"],
            "width": session["width"],
            "height": session["height"],
            "format": session["format"],
            "thumbnail": f"/api/thumbnail/{image_id}",
            "results": session.get("results"),
        })
    return {"images": images}


@app.delete("/api/images")
async def clear_images():
    """Clear all uploaded images."""
    # Clean up all thumbnail and preview files
    for session in _sessions.values():
        thumb_path = session.get("thumb_path")
        if thumb_path and thumb_path.exists():
            thumb_path.unlink(missing_ok=True)
        preview_path = session.get("preview_path")
        if preview_path and preview_path.exists():
            preview_path.unlink(missing_ok=True)

    _sessions.clear()
    return {"status": "cleared"}


def create_app() -> FastAPI:
    """Factory function to create the app."""
    return app
