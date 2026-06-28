"""Build the inline chat embed for a generated image.

Luna renders a tool result's `embed_iframe` (a self-contained HTML document)
directly in the conversation — the same hook `plugin-charts` / `plugin-giphy`
use. The image is referenced by its served URL (a root-relative path that the
sandboxed `srcdoc` iframe resolves against the parent page origin), so no heavy
base64 ever lands in the document or the model's context.
"""

from __future__ import annotations

import html as _html

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
    <a href="{url}" target="_blank" rel="noopener">
      <img src="{url}" alt="{alt}" loading="lazy">
    </a>
  </div>
  <div class="meta">
    <span class="caption">{caption}</span>
    <span class="badge">{badge}</span>
  </div>
</body>
</html>"""


def render_image_embed(image_url: str, *, prompt: str = "", model_label: str = "") -> str:
    """Return a self-contained HTML document that shows the image inline."""
    return _TEMPLATE.format(
        url=_html.escape(image_url, quote=True),
        alt=_html.escape(prompt or "Generated image", quote=True),
        caption=_html.escape((prompt or "")[:240]),
        badge=_html.escape(model_label or "image"),
    )
