"""plugin-image-gen — let the agent create and edit images, shown inline in chat.

Authored against `luna_sdk` ONLY (never `import luna.*`). The agent describes an
image, this plugin renders it with the best available model and returns an
`embed_iframe` that Luna shows inline in the conversation (same render hook
`plugin-charts` / `plugin-giphy` use).

Models (pick with `model=`; default "best"):
  - nano-banana-pro  → Gemini 3 Pro Image  (default — top quality, clean text)
  - nano-banana      → Gemini 2.5 Flash Image (cheaper/faster)
  - gpt-image-1      → OpenAI GPT Image
  - flux-pro         → FLUX 1.1 [pro]  (Black Forest Labs)
  - flux-pro-ultra   → FLUX 1.1 [pro] ultra

Tools:
  - generate_image      — text → image, shown inline
  - edit_image          — restyle/modify a reference image (Gemini / GPT Image)
  - list_image_models   — list models + which providers have a key configured

API key resolution per provider (first hit wins):
  vault `<provider>_api_key` → env `LUNA_<PROVIDER>_API_KEY` → native env
  (`GEMINI_API_KEY` / `OPENAI_API_KEY` / `BFL_API_KEY`).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from luna_sdk import CredentialSlot, LunaPlugin, PluginContext, PluginManifest, ToolDef

from . import providers, storage
from .render import render_image_embed

log = logging.getLogger("plugin-image-gen")

# provider -> (vault credential name, LUNA_ key env, LUNA_ base-url env,
#              native key env names, native base-url env names)
_KEYS = {
    "gemini": ("gemini_api_key", "LUNA_GEMINI_API_KEY", "LUNA_GEMINI_BASE_URL", ("GEMINI_API_KEY", "GOOGLE_API_KEY"), ("GEMINI_BASE_URL",)),
    "openai": ("openai_api_key", "LUNA_OPENAI_API_KEY", "LUNA_OPENAI_BASE_URL", ("OPENAI_API_KEY",), ("OPENAI_BASE_URL",)),
    "flux": ("bfl_api_key", "LUNA_BFL_API_KEY", "LUNA_BFL_BASE_URL", ("BFL_API_KEY",), ("BFL_BASE_URL",)),
}

_MODEL_ENUM = ["best"] + list(providers.MODELS.keys())

_GENERATE_DEF = ToolDef(
    name="generate_image",
    description=(
        "Generate an image from a text prompt and show it INLINE in the chat. "
        "Use whenever the owner asks to create / make / draw / design an image, "
        "illustration, logo, icon, poster, concept art, photo, texture, or any "
        "visual. Write a vivid, specific `prompt` (subject, style, colors, mood, "
        "composition). Defaults to the best model (Nano Banana Pro). The image "
        "renders directly in the conversation — keep accompanying text short."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Vivid description of the image to create."},
            "model": {
                "type": "string",
                "enum": _MODEL_ENUM,
                "description": "Model to use (default 'best' = Nano Banana Pro).",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": list(providers.VALID_ASPECTS),
                "description": "Aspect ratio (default 1:1).",
            },
        },
        "required": ["prompt"],
    },
    policy="ask",
    risk_level="low",
    timeout_seconds=180,
)

_EDIT_DEF = ToolDef(
    name="edit_image",
    description=(
        "Edit, restyle, or modify an existing image and show the result INLINE. "
        "Provide `prompt` (what to change) plus the source as `image_path` (a "
        "local path, e.g. the `file_path` returned by a previous generate_image "
        "call) or `image_url` (an http(s) image). Uses Nano Banana Pro by default; "
        "GPT Image also supported. Great for 'make it night', 'add a hat', "
        "'turn this into a watercolor', 'remove the background'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The change/restyle to apply."},
            "image_path": {"type": "string", "description": "Local source image path (e.g. a prior file_path)."},
            "image_url": {"type": "string", "description": "http(s) URL of the source image (alternative to image_path)."},
            "model": {
                "type": "string",
                "enum": ["best", "nano-banana-pro", "nano-banana", "gpt-image-1"],
                "description": "Edit-capable model (default 'best').",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": list(providers.VALID_ASPECTS),
                "description": "Output aspect ratio (default 1:1).",
            },
        },
        "required": ["prompt"],
    },
    policy="ask",
    risk_level="low",
    timeout_seconds=180,
)

_LIST_DEF = ToolDef(
    name="list_image_models",
    description=(
        "List the available image models, their providers, capabilities, and "
        "which ones currently have an API key configured. Call this when unsure "
        "which model to use or when generation fails with a missing-key error."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    policy="auto_approve",
    risk_level="low",
)


class ImageGenPlugin(LunaPlugin):
    manifest = PluginManifest(
        name="plugin-image-gen",
        shown_name="Image Generation",
        icon="image",
        image="assets/icon.png",
        version="0.3.6",
        description=(
            "Generate and edit images inline in chat with the best models — "
            "Nano Banana Pro (Gemini), GPT Image, and FLUX. Built on luna_sdk v0."
        ),
        tools=[_GENERATE_DEF, _EDIT_DEF, _LIST_DEF],
        routes_module="routes",
        # Consume the storage capability so generated images also land in Files
        # (visible/durable). The inline preview still uses the plugin's own public
        # route; Files is the findable copy.
        capabilities=["storage"],
    )

    def __init__(self) -> None:
        self._ctx: PluginContext | None = None

    def credential_slots(self) -> list[CredentialSlot]:
        # env_base_url_var on each slot is the signal the cloud control plane +
        # UI read to mark a provider as proxy-provisionable (007-provider-base-url).
        return [
            CredentialSlot(
                slug=provider,
                credential_name=vault_name,
                env_key_var=key_var,
                env_base_url_var=base_var,
                owner=self.manifest.name,
            )
            for provider, (vault_name, key_var, base_var, _nk, _nb) in _KEYS.items()
        ]

    async def _api_key(self, provider: str) -> str | None:
        """vault `<provider>_api_key` → LUNA_ env → native env. None if unset.

        In cloud proxy mode the env value is the opaque gateway token, which the
        plugin treats exactly like a key.
        """
        vault_name, key_var, _base_var, native, _nb = _KEYS[provider]
        ctx = self._ctx
        if ctx is not None and getattr(ctx, "vault", None) is not None:
            try:
                cred = await ctx.vault.get_credential(vault_name)
                if (cred.value or "").strip():
                    return cred.value.strip()
            except KeyError:
                pass
            except Exception as exc:  # noqa: BLE001 — a vault hiccup must not block generation
                log.warning("image-gen: vault read failed for %s: %s", vault_name, exc)
        if ctx is not None and getattr(ctx, "get_env", None) is not None:
            val = (ctx.get_env(key_var) or "").strip()
            if val:
                return val
        for name in native:
            val = (os.environ.get(name) or "").strip()
            if val:
                return val
        return None

    def _base_url(self, provider: str) -> str | None:
        """`LUNA_<PROVIDER>_BASE_URL` → native base env. None = real upstream.

        When set it points at `{gateway}/proxy/<provider>` so the real key never
        lands on the tenant machine; otherwise the provider's real API is used.
        """
        _vault, _key, base_var, _nk, native_bases = _KEYS[provider]
        ctx = self._ctx
        if ctx is not None and getattr(ctx, "get_env", None) is not None:
            v = (ctx.get_env(base_var) or "").strip()
            if v:
                return v
        for name in native_bases:
            v = (os.environ.get(name) or "").strip()
            if v:
                return v
        return None

    async def on_load(self, ctx: PluginContext) -> None:
        self._ctx = ctx

        async def _generate_image(prompt: str, model: str = "best", aspect_ratio: str = "1:1") -> str:
            m = providers.resolve_model(model)
            if m is None:
                return json.dumps({"error": "unknown model", "detail": f"No such model '{model}'. Call list_image_models."})
            key = await self._api_key(m.provider)
            if not key:
                return json.dumps(self._missing_key_result(m.provider))
            result = await providers.generate(
                m, prompt, aspect=aspect_ratio, api_key=key, base_url=self._base_url(m.provider),
            )
            return await self._finish(result, m, prompt, aspect_ratio)

        async def _edit_image(
            prompt: str,
            image_path: str | None = None,
            image_url: str | None = None,
            model: str = "best",
            aspect_ratio: str = "1:1",
        ) -> str:
            m = providers.resolve_model(model)
            if m is None or not m.supports_edit:
                m = providers.MODELS[providers.DEFAULT_MODEL]
            source = await self._load_source(image_path, image_url)
            if "error" in source:
                return json.dumps(source)
            key = await self._api_key(m.provider)
            if not key:
                return json.dumps(self._missing_key_result(m.provider))
            result = await providers.edit(
                m, prompt, (source["bytes"], source["mime"]), aspect=aspect_ratio,
                api_key=key, base_url=self._base_url(m.provider),
            )
            return await self._finish(result, m, prompt, aspect_ratio)

        async def _list_image_models() -> str:
            out = []
            for m in providers.MODELS.values():
                out.append({
                    "model": m.key,
                    "provider": m.provider,
                    "label": m.label,
                    "supports_edit": m.supports_edit,
                    "key_configured": bool(await self._api_key(m.provider)),
                    "default": m.key == providers.DEFAULT_MODEL,
                })
            return json.dumps({"models": out, "aspect_ratios": list(providers.VALID_ASPECTS)})

        ctx.tool_registry.register(self.manifest.name, _GENERATE_DEF, _generate_image)
        ctx.tool_registry.register(self.manifest.name, _EDIT_DEF, _edit_image)
        ctx.tool_registry.register(self.manifest.name, _LIST_DEF, _list_image_models)
        log.info("image-gen.tools_registered: generate_image, edit_image, list_image_models")

    # ---- helpers -------------------------------------------------------- #
    def _missing_key_result(self, provider: str) -> dict[str, Any]:
        vault_name, key_var, _base_var, _native, _nb = _KEYS[provider]
        return {
            "error": f"{provider} api key missing",
            "detail": (
                f"No API key for '{provider}'. Add the vault credential "
                f"`{vault_name}` (Settings → Vault) or set env `{key_var}`, then retry. "
                f"Use list_image_models to see what is configured."
            ),
        }

    async def _finish(self, result: dict[str, Any], model: providers.Model, prompt: str, aspect_ratio: str) -> str:
        """Persist the bytes, build the inline embed, return a SMALL JSON string.

        Bytes are written twice on purpose: to this plugin's own store (powers the
        public route the sandboxed inline `<img>` loads — the Files `/read` URL is
        auth-gated and can't render in the chat iframe), and, when a storage
        provider is enabled, into Files under `images/` so the user can actually
        find and keep the image.
        """
        if "error" in result:
            return json.dumps(result)
        data = result["image_bytes"]
        mime = result.get("mime", "image/png")
        saved = storage.save_image(data, mime)  # inline-render copy (public route)
        files_ref = await self._save_to_files(data, mime, str(saved["id"]))

        payload: dict[str, Any] = {
            "ok": True,
            "provider": model.provider,
            "model": model.key,
            "model_label": model.label,
            "prompt": prompt[:500],
            "aspect_ratio": providers.clean_aspect(aspect_ratio),
            "image_url": saved["url"],
            "file_path": saved["path"],
            "mime": saved["mime"],
            "bytes": saved["bytes"],
        }
        if files_ref:
            # Tell the agent (and the user, via the caption) exactly where it landed.
            payload["saved_to_files"] = files_ref
            payload["files_note"] = f"Saved to Files → {files_ref}"
        payload["embed_iframe"] = render_image_embed(
            str(saved["url"]), prompt=prompt, model_label=model.label, saved_to=files_ref or "",
        )
        return json.dumps(payload)

    def _storage_provider(self) -> Any | None:
        """Resolve the Files StorageProvider.

        plugin-files registers under the provider-registry key ``"storage"``
        (`manifest.provider="storage"`), so that is the canonical access path on
        current cores — there is no ``ctx.storage`` shortcut yet. We prefer the
        registry and fall back to ``ctx.storage`` only for a future core that adds
        the convenience property. Resolved per call (never cached at on_load) so
        we see plugin-files whenever it registers, regardless of load order.
        """
        ctx = self._ctx
        if ctx is None:
            return None
        registry = getattr(ctx, "provider_registry", None)
        if registry is not None:
            try:
                if registry.has("storage"):
                    return registry.get("storage")
            except Exception as exc:  # noqa: BLE001 — registry hiccup must not block the render
                log.warning("image-gen: storage provider lookup failed: %s", exc)
        return getattr(ctx, "storage", None)

    async def _save_to_files(self, data: bytes, mime: str, name: str) -> str | None:
        """Best-effort copy into the Files storage provider. Returns the Files
        ref (e.g. `images/<name>`) or None when no storage provider is enabled."""
        provider = self._storage_provider()
        if provider is None:
            return None
        ref = f"images/{name}"
        try:
            stored = await provider.save(data, filename=ref, media_type=mime)
            return getattr(stored, "ref", None) or ref
        except Exception as exc:  # noqa: BLE001 — Files copy is a nicety, never block the render
            log.warning("image-gen: could not save to Files (%s): %s", ref, exc)
            return None

    async def _load_source(self, image_path: str | None, image_url: str | None) -> dict[str, Any]:
        if image_path:
            src = storage.read_source(image_path)
            if src is None:
                return {"error": "source not found", "detail": f"No readable image at {image_path}."}
            return {"bytes": src[0], "mime": src[1]}
        url = (image_url or "").strip()
        if url.startswith(("http://", "https://")):
            try:
                async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": providers.USER_AGENT}) as cli:
                    resp = await cli.get(url)
                    if resp.status_code >= 400:
                        return {"error": "source fetch failed", "detail": f"HTTP {resp.status_code} fetching {url}."}
                    mime = resp.headers.get("content-type", "image/png").split(";")[0]
                    return {"bytes": resp.content, "mime": mime}
            except httpx.HTTPError as exc:
                return {"error": "source fetch failed", "detail": str(exc)}
        return {"error": "no source image", "detail": "Provide image_path (local) or image_url (http) to edit."}
