"""Image-generation clients for the best models available — one tiny seam each.

Pure logic against `httpx`; imports nothing from `luna_sdk` so it unit-tests
anywhere. Every public function NEVER raises — it returns either
``{"image_bytes": <bytes>, "mime": <str>}`` on success or an ``{"error": ...,
"detail": ...}`` dict the tool layer relays to the agent.

Three providers behind one catalog:
  - gemini  → Nano Banana Pro / Nano Banana (Google generateContent REST)
  - openai  → GPT Image (images/generations + images/edits)
  - flux    → FLUX 1.1 [pro] / [pro] ultra (Black Forest Labs async API)

The caller resolves API keys (vault/env) and passes them in; this module only
takes a key per call.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_TIMEOUT = 120.0
USER_AGENT = "Luna/1.0 (AI Agent; +https://github.com/huemorgan/plugin-image-gen)"

GEMINI_ROOT = "https://generativelanguage.googleapis.com/v1beta"
OPENAI_ROOT = "https://api.openai.com/v1"
BFL_ROOT = "https://api.bfl.ai"

VALID_ASPECTS = ("1:1", "16:9", "9:16", "4:3", "3:2", "21:9")


@dataclass(frozen=True)
class Model:
    """One selectable image model and what it can do."""

    key: str          # friendly name the agent picks (e.g. "nano-banana-pro")
    provider: str     # "gemini" | "openai" | "flux"
    api_id: str       # the provider's own model id
    label: str        # human label for UI / results
    supports_edit: bool
    supports_aspect: bool


# The catalog, best-first. `generate_image(model="best")` resolves to the head.
MODELS: dict[str, Model] = {
    "nano-banana-pro": Model(
        "nano-banana-pro", "gemini", "gemini-3-pro-image",
        "Nano Banana Pro (Gemini 3 Pro Image)", supports_edit=True, supports_aspect=True,
    ),
    "nano-banana": Model(
        "nano-banana", "gemini", "gemini-2.5-flash-image",
        "Nano Banana (Gemini 2.5 Flash Image)", supports_edit=True, supports_aspect=True,
    ),
    "gpt-image-1": Model(
        "gpt-image-1", "openai", "gpt-image-1",
        "GPT Image 1 (OpenAI)", supports_edit=True, supports_aspect=True,
    ),
    "flux-pro": Model(
        "flux-pro", "flux", "flux-pro-1.1",
        "FLUX 1.1 [pro] (Black Forest Labs)", supports_edit=False, supports_aspect=True,
    ),
    "flux-pro-ultra": Model(
        "flux-pro-ultra", "flux", "flux-pro-1.1-ultra",
        "FLUX 1.1 [pro] ultra (Black Forest Labs)", supports_edit=False, supports_aspect=True,
    ),
}

# The first key is the global default ("best").
DEFAULT_MODEL = "nano-banana-pro"

# Friendly synonyms the agent might use.
ALIASES: dict[str, str] = {
    "best": DEFAULT_MODEL,
    "default": DEFAULT_MODEL,
    "gemini": "nano-banana-pro",
    "google": "nano-banana-pro",
    "nano-banana-2": "nano-banana-pro",
    "openai": "gpt-image-1",
    "dalle": "gpt-image-1",
    "dall-e": "gpt-image-1",
    "gpt": "gpt-image-1",
    "gpt-image": "gpt-image-1",
    "flux": "flux-pro",
    "flux-ultra": "flux-pro-ultra",
}

PROVIDER_ORDER = ("gemini", "openai", "flux")


def resolve_model(name: str | None) -> Model | None:
    """Map a friendly name/alias to a Model, or None if unknown."""
    key = (name or "best").strip().lower()
    key = ALIASES.get(key, key)
    return MODELS.get(key)


def clean_aspect(aspect: str | None) -> str:
    a = (aspect or "1:1").strip()
    return a if a in VALID_ASPECTS else "1:1"


def _aspect_to_size(aspect: str) -> str:
    """OpenAI accepts only square / portrait / landscape — pick the closest."""
    a = clean_aspect(aspect)
    if a in ("16:9", "3:2", "21:9"):
        return "1536x1024"
    if a in ("9:16",):
        return "1024x1536"
    if a in ("4:3",):
        return "1024x1024"
    return "1024x1024"


def _aspect_to_wh(aspect: str) -> tuple[int, int]:
    """FLUX [pro] takes width/height (multiples of 32, ~1MP)."""
    return {
        "1:1": (1024, 1024),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "4:3": (1184, 880),
        "3:2": (1248, 832),
        "21:9": (1440, 608),
    }.get(clean_aspect(aspect), (1024, 1024))


def _client(client: httpx.AsyncClient | None) -> tuple[httpx.AsyncClient, bool]:
    if client is not None:
        return client, False
    return httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers={"User-Agent": USER_AGENT}), True


def _missing_key(provider: str) -> dict[str, str]:
    env = {"gemini": "LUNA_GEMINI_API_KEY", "openai": "LUNA_OPENAI_API_KEY", "flux": "LUNA_BFL_API_KEY"}[provider]
    vault = {"gemini": "gemini_api_key", "openai": "openai_api_key", "flux": "bfl_api_key"}[provider]
    return {
        "error": f"{provider} api key missing",
        "detail": (
            f"No API key for the '{provider}' provider. Store it as the vault "
            f"credential `{vault}` (Settings → Vault) or set env `{env}`, then retry."
        ),
    }


# --------------------------------------------------------------------------- #
# Gemini (Nano Banana / Nano Banana Pro)
# --------------------------------------------------------------------------- #
async def _gemini_generate(
    model: Model,
    prompt: str,
    *,
    aspect: str | None,
    refs: list[tuple[bytes, str]] | None,
    api_key: str,
    client: httpx.AsyncClient | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for data, mime in refs or []:
        parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}})
    gen_cfg: dict[str, Any] = {"responseModalities": ["IMAGE"]}
    if aspect:
        gen_cfg["imageConfig"] = {"aspectRatio": clean_aspect(aspect)}
    body = {"contents": [{"parts": parts}], "generationConfig": gen_cfg}
    root = (base_url or GEMINI_ROOT).rstrip("/")
    url = f"{root}/models/{model.api_id}:generateContent?key={api_key}"

    cli, owns = _client(client)
    try:
        resp = await cli.post(url, json=body, headers={"Content-Type": "application/json"})
        if resp.status_code in (401, 403):
            return _missing_key("gemini")
        if resp.status_code == 429:
            return {"error": "gemini rate limited", "detail": "Gemini rate limit hit; retry shortly."}
        if resp.status_code >= 400:
            return {"error": "gemini request failed", "detail": _trim(resp.text), "status": resp.status_code}
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        return {"error": "gemini request failed", "detail": str(exc)}
    finally:
        if owns:
            await cli.aclose()

    cands = data.get("candidates") or []
    if not cands:
        fb = data.get("promptFeedback") or {}
        return {"error": "no image", "detail": f"Gemini returned no image. {json.dumps(fb)[:300]}".strip()}
    for part in cands[0].get("content", {}).get("parts", []):
        inl = part.get("inlineData") or part.get("inline_data")
        if inl and inl.get("data"):
            return {"image_bytes": base64.b64decode(inl["data"]), "mime": inl.get("mimeType") or inl.get("mime_type") or "image/png"}
    texts = " ".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))
    return {"error": "no image", "detail": "Model returned text, not an image: " + _trim(texts)}


# --------------------------------------------------------------------------- #
# OpenAI (GPT Image)
# --------------------------------------------------------------------------- #
async def _openai_generate(
    model: Model,
    prompt: str,
    *,
    aspect: str | None,
    api_key: str,
    client: httpx.AsyncClient | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    body = {"model": model.api_id, "prompt": prompt, "n": 1, "size": _aspect_to_size(aspect or "1:1")}
    root = (base_url or OPENAI_ROOT).rstrip("/")
    cli, owns = _client(client)
    try:
        resp = await cli.post(
            f"{root}/images/generations",
            json=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        return _openai_parse(resp)
    except (httpx.HTTPError, ValueError) as exc:
        return {"error": "openai request failed", "detail": str(exc)}
    finally:
        if owns:
            await cli.aclose()


async def _openai_edit(
    model: Model,
    prompt: str,
    *,
    image: tuple[bytes, str],
    aspect: str | None,
    api_key: str,
    client: httpx.AsyncClient | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    data, mime = image
    files = {"image": ("source.png", data, mime or "image/png")}
    form = {"model": model.api_id, "prompt": prompt, "n": "1", "size": _aspect_to_size(aspect or "1:1")}
    root = (base_url or OPENAI_ROOT).rstrip("/")
    cli, owns = _client(client)
    try:
        resp = await cli.post(
            f"{root}/images/edits",
            data=form,
            files=files,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        return _openai_parse(resp)
    except (httpx.HTTPError, ValueError) as exc:
        return {"error": "openai request failed", "detail": str(exc)}
    finally:
        if owns:
            await cli.aclose()


def _openai_parse(resp: httpx.Response) -> dict[str, Any]:
    if resp.status_code in (401, 403):
        return _missing_key("openai")
    if resp.status_code == 429:
        return {"error": "openai rate limited", "detail": "OpenAI rate limit / quota hit; retry shortly."}
    if resp.status_code >= 400:
        return {"error": "openai request failed", "detail": _trim(resp.text), "status": resp.status_code}
    data = resp.json()
    items = data.get("data") or []
    if not items or not items[0].get("b64_json"):
        return {"error": "no image", "detail": "OpenAI returned no image data."}
    return {"image_bytes": base64.b64decode(items[0]["b64_json"]), "mime": "image/png"}


# --------------------------------------------------------------------------- #
# FLUX (Black Forest Labs) — async submit + poll
# --------------------------------------------------------------------------- #
async def _flux_generate(
    model: Model,
    prompt: str,
    *,
    aspect: str | None,
    api_key: str,
    client: httpx.AsyncClient | None = None,
    base_url: str | None = None,
    poll_interval: float = 1.5,
    max_polls: int = 60,
) -> dict[str, Any]:
    root = (base_url or BFL_ROOT).rstrip("/")
    headers = {"x-key": api_key, "accept": "application/json", "Content-Type": "application/json"}
    if model.api_id.endswith("ultra"):
        body: dict[str, Any] = {"prompt": prompt, "aspect_ratio": clean_aspect(aspect)}
    else:
        w, h = _aspect_to_wh(aspect or "1:1")
        body = {"prompt": prompt, "width": w, "height": h}

    cli, owns = _client(client)
    try:
        resp = await cli.post(f"{root}/v1/{model.api_id}", json=body, headers=headers)
        if resp.status_code in (401, 403):
            return _missing_key("flux")
        if resp.status_code == 429:
            return {"error": "flux rate limited", "detail": "FLUX rate limit hit; retry shortly."}
        if resp.status_code >= 400:
            return {"error": "flux request failed", "detail": _trim(resp.text), "status": resp.status_code}
        submit = resp.json()
        polling_url = submit.get("polling_url")
        if not polling_url:
            return {"error": "flux request failed", "detail": "No polling_url in FLUX response."}

        sample_url: str | None = None
        for _ in range(max(1, max_polls)):
            poll = await cli.get(polling_url, headers={"x-key": api_key, "accept": "application/json"})
            if poll.status_code >= 400:
                return {"error": "flux poll failed", "detail": _trim(poll.text), "status": poll.status_code}
            pdata = poll.json()
            status = (pdata.get("status") or "").strip()
            if status == "Ready":
                sample_url = (pdata.get("result") or {}).get("sample")
                break
            if status in ("Error", "Failed"):
                return {"error": "flux generation failed", "detail": _trim(json.dumps(pdata))}
            if status in ("Request Moderated", "Content Moderated"):
                return {"error": "flux moderated", "detail": "FLUX blocked this prompt or its output by moderation."}
            await asyncio.sleep(poll_interval)
        if not sample_url:
            return {"error": "flux timeout", "detail": "FLUX did not finish in time; try again."}

        img = await cli.get(sample_url)
        if img.status_code >= 400:
            return {"error": "flux download failed", "detail": f"HTTP {img.status_code} fetching the result image."}
        return {"image_bytes": img.content, "mime": img.headers.get("content-type", "image/png").split(";")[0]}
    except (httpx.HTTPError, ValueError) as exc:
        return {"error": "flux request failed", "detail": str(exc)}
    finally:
        if owns:
            await cli.aclose()


# --------------------------------------------------------------------------- #
# Public dispatch
# --------------------------------------------------------------------------- #
async def generate(
    model: Model,
    prompt: str,
    *,
    aspect: str | None = None,
    api_key: str,
    client: httpx.AsyncClient | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Generate one image with `model`. Returns image bytes or an error dict.

    `base_url` overrides the provider's upstream — set by the caller to
    `{gateway}/proxy/<provider>` for cloud key-provisioning (the resolved
    `api_key` is then the opaque gateway token). When None the real upstream is
    used (BYO key / local dev).
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return {"error": "empty prompt", "detail": "Describe the image you want."}
    if not api_key:
        return _missing_key(model.provider)
    if model.provider == "gemini":
        return await _gemini_generate(model, prompt, aspect=aspect, refs=None, api_key=api_key, client=client, base_url=base_url)
    if model.provider == "openai":
        return await _openai_generate(model, prompt, aspect=aspect, api_key=api_key, client=client, base_url=base_url)
    if model.provider == "flux":
        return await _flux_generate(model, prompt, aspect=aspect, api_key=api_key, client=client, base_url=base_url)
    return {"error": "unknown provider", "detail": model.provider}


async def edit(
    model: Model,
    prompt: str,
    image: tuple[bytes, str],
    *,
    aspect: str | None = None,
    api_key: str,
    client: httpx.AsyncClient | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Edit/restyle a reference image. Only gemini + openai support this."""
    prompt = (prompt or "").strip()
    if not prompt:
        return {"error": "empty prompt", "detail": "Describe the change you want."}
    if not api_key:
        return _missing_key(model.provider)
    if model.provider == "gemini":
        return await _gemini_generate(model, prompt, aspect=aspect, refs=[image], api_key=api_key, client=client, base_url=base_url)
    if model.provider == "openai":
        return await _openai_edit(model, prompt, image=image, aspect=aspect, api_key=api_key, client=client, base_url=base_url)
    return {"error": "edit unsupported", "detail": f"{model.label} cannot edit images; use nano-banana-pro or gpt-image-1."}


def _trim(text: str, n: int = 400) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:n]
