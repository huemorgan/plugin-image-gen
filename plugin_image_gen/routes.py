"""plugin-image-gen routes — serve generated images to the chat embed.

The chat renders each result's `embed_iframe`, whose `<img src>` points at
`/api/p/plugin-image-gen/file/<id>`. That iframe is sandboxed (`allow-scripts`
only, opaque origin), so this GET is intentionally UNAUTHENTICATED — a cookie
wouldn't be sent anyway. Ids are random (uuid4 hex), so the route is not
enumerable, and `storage.resolve` rejects any path traversal.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from . import storage

_MEDIA = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}


def register_routes(app, ctx):
    router = APIRouter(prefix="/api/p/plugin-image-gen", tags=["image-gen"])

    @router.get("/file/{name}")
    async def serve_image(name: str):
        path = storage.resolve(name)
        if path is None:
            raise HTTPException(404, "Image not found")
        media = _MEDIA.get(path.suffix.lower().lstrip("."), "application/octet-stream")
        return FileResponse(
            str(path),
            media_type=media,
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    app.include_router(router)
