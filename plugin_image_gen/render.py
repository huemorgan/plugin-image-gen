"""Build the inline chat embed for a generated image.

Luna renders a tool result's `embed_iframe` (a self-contained HTML document)
directly in the conversation — the same hook `plugin-charts` / `plugin-giphy`
use. The image is referenced by its served URL, so no heavy base64 ever lands in
the document or the model's context.

Path-prefix safety: the chat iframe is `sandbox="allow-scripts"` (an opaque
origin — it CANNOT read `window.parent`), and Luna can be hosted behind a mount
prefix (luna-service serves each tenant under `/a/<slug>/...`). A root-absolute
`/api/...` src resolves to the host root and 404s there.

A plain relative URL is NOT enough either: the chat URL is `<mount>/chat/<id>`
(two levels deep — see Luna's `Shell.tsx` router), so `api/p/...` would resolve
to `<mount>/chat/api/p/...` and 404. We can't read the mount from JS (opaque
origin blocks `window.parent`), but an `about:srcdoc` document inherits the
PARENT's base URL, so `document.baseURI` is the full chat URL. We strip the
trailing SPA route segment (`/chat/<id>`, `/settings/...`, `/p/<id>`,
`/approvals`) to recover `<origin><mount>` and build an absolute image URL that
is correct at any route depth and behind any prefix. The relative URL stays as a
no-JS fallback.
"""

from __future__ import annotations

import html as _html
import json as _json


def _embed_src(image_url: str) -> str:
    """Make the served URL relative so the sandboxed srcdoc iframe resolves it
    against the parent's mount prefix instead of the host root."""
    return image_url.lstrip("/") if image_url.startswith("/") else image_url


def _mount_script(rel_url: str) -> str:
    """Inline JS that rewrites the image src/href to an absolute, mount-correct
    URL derived from the parent page's base URL. No-op if anything fails (the
    static relative `src` remains as the fallback)."""
    rel_js = _json.dumps(rel_url)
    return (
        "<script>(function(){try{"
        f"var rel={rel_js};"
        "var u=new URL(document.baseURI);"
        "var mount=u.pathname.replace(/\\/(chat|settings|approvals|p)(\\/[^?#]*)?$/,'');"
        "if(mount==='/')mount='';"
        "mount=mount.replace(/\\/$/,'');"
        "var abs=u.origin+mount+'/'+rel.replace(/^\\//,'');"
        "var img=document.getElementById('img');"
        "var lnk=document.getElementById('lnk');"
        "if(img)img.src=abs;"
        "if(lnk)lnk.href=abs;"
        "}catch(e){}})();</script>"
    )

_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0f0f1a;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }}
  .img-wrap {{
    width: 100%;
    max-width: 512px;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 6px 24px rgba(0,0,0,0.45);
    background: #1a1a2e;
  }}
  .img-wrap img {{ display: block; width: 100%; height: auto; }}
  .meta {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    max-width: 512px;
    gap: 8px;
  }}
  .caption {{ color: #c7c7d9; font-size: 12px; line-height: 1.35; flex: 1; }}
  .saved {{ color: #6f6f86; font-size: 11px; }}
  .badge {{
    color: #8a8aa3;
    font-size: 11px;
    font-weight: 600;
    white-space: nowrap;
    border: 1px solid #2a2a44;
    border-radius: 999px;
    padding: 2px 8px;
  }}
</style>
</head>
<body>
  <div class="img-wrap">
    <a id="lnk" href="{url}" target="_blank" rel="noopener">
      <img id="img" src="{url}" alt="{alt}" loading="lazy">
    </a>
  </div>
  <div class="meta">
    <span class="caption">{caption}</span>
    <span class="badge">{badge}</span>
  </div>
  {saved}
  {script}
</body>
</html>"""

_SAVED_LINE = '<span class="saved">Saved to Files \u2192 {ref}</span>'


def render_image_embed(
    image_url: str, *, prompt: str = "", model_label: str = "", saved_to: str = ""
) -> str:
    """Return a self-contained HTML document that shows the image inline.

    When ``saved_to`` is given (the Files ref, e.g. ``images/<name>``), a small
    line tells the user where to find the saved copy.
    """
    saved = _SAVED_LINE.format(ref=_html.escape(saved_to)) if saved_to else ""
    rel = _embed_src(image_url)
    return _TEMPLATE.format(
        url=_html.escape(rel, quote=True),
        alt=_html.escape(prompt or "Generated image", quote=True),
        caption=_html.escape((prompt or "")[:240]),
        badge=_html.escape(model_label or "image"),
        saved=saved,
        script=_mount_script(rel),
    )
