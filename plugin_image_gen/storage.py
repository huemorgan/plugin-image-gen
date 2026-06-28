"""On-disk store for generated images, shared by the tool layer and the route.

Why a disk store instead of a base64 data URI: a tool result is fed back to the
model verbatim on the next turn, so inlining ~1MB of base64 would blow the
context window. Instead we write the bytes once, hand the model a tiny URL
(`/api/p/plugin-image-gen/file/<id>`), and let the chat embed `<img>` load it
from that route. Files live under a temp dir so they survive a server restart
(until the OS clears temp), which keeps already-rendered chat images working.

No `luna_sdk` import here — pure stdlib so it unit-tests anywhere.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

URL_PREFIX = "/api/p/plugin-image-gen/file"

_EXT_BY_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}


def store_dir() -> Path:
    """Where images are written. Overridable via `LUNA_IMAGE_GEN_DIR`."""
    override = os.environ.get("LUNA_IMAGE_GEN_DIR")
    base = Path(override) if override else Path(tempfile.gettempdir()) / "luna-image-gen"
    return base


def _ext_for(mime: str) -> str:
    return _EXT_BY_MIME.get((mime or "").lower().split(";")[0].strip(), "png")


def save_image(data: bytes, mime: str = "image/png") -> dict[str, str | int]:
    """Persist image bytes and return its id, absolute path, served URL, mime, size."""
    d = store_dir()
    d.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.{_ext_for(mime)}"
    path = d / name
    path.write_bytes(data)
    return {
        "id": name,
        "path": str(path),
        "url": f"{URL_PREFIX}/{name}",
        "mime": (mime or "image/png").split(";")[0],
        "bytes": len(data),
    }


def resolve(name: str) -> Path | None:
    """Map a served file name back to a real path, rejecting traversal."""
    if not name or "/" in name or "\\" in name or ".." in name:
        return None
    path = store_dir() / name
    if path.is_file():
        return path
    return None


def read_source(path_or_bytes: str | bytes) -> tuple[bytes, str] | None:
    """Load a local source image (for edits) by absolute path. Returns (bytes, mime)."""
    if isinstance(path_or_bytes, bytes):
        return path_or_bytes, "image/png"
    p = Path(path_or_bytes)
    if not p.is_file():
        return None
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(
        p.suffix.lower().lstrip("."), "image/png"
    )
    return p.read_bytes(), mime
