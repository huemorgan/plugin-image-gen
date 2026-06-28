"""Unit tests for plugin-image-gen — the inner loop, no Luna runtime needed.

Manifest tests read the TOML data contract directly. Provider logic is exercised
via httpx.MockTransport (no real network). Tool-wiring tests load the plugin
against a fake context and monkeypatch the provider layer.

Run: `pip install -e ".[dev]" && pytest`
"""

from __future__ import annotations

import base64
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from plugin_image_gen import ImageGenPlugin
from plugin_image_gen import providers, storage
from plugin_image_gen.render import render_image_embed

PKG = Path(__file__).resolve().parents[1] / "plugin_image_gen"
PNG = b"\x89PNG\r\n\x1a\n-fake-bytes"


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _manifest() -> dict:
    return tomllib.loads((PKG / "luna-plugin.toml").read_text())


# ---------------- manifest / data contract ----------------
def test_identity() -> None:
    m = _manifest()
    assert m["name"] == "plugin-image-gen"
    assert m["entry"] == "plugin_image_gen"
    assert m["sdk_version"] == "0"
    assert m["routes_module"] == "routes"


def test_tool_count_matches_requires() -> None:
    m = _manifest()
    assert len(m["tools"]) == m["requires"]["tools"] == 3
    names = {t["name"] for t in m["tools"]}
    assert names == {"generate_image", "edit_image", "list_image_models"}


def test_manifest_and_code_versions_agree() -> None:
    toml_version = _manifest()["version"]
    init_src = (PKG / "__init__.py").read_text()
    code_version = re.search(r'version="([^"]+)"', init_src).group(1)
    assert toml_version == code_version


def test_manifest_matches_code_identity() -> None:
    assert ImageGenPlugin.manifest.name == _manifest()["name"]
    assert ImageGenPlugin.manifest.version == _manifest()["version"]


def test_credential_slots_advertise_base_url_vars() -> None:
    slots = {s.slug: s for s in ImageGenPlugin().credential_slots()}
    assert set(slots) == {"gemini", "openai", "flux"}
    assert slots["gemini"].env_key_var == "LUNA_GEMINI_API_KEY"
    assert slots["gemini"].env_base_url_var == "LUNA_GEMINI_BASE_URL"
    assert slots["openai"].env_base_url_var == "LUNA_OPENAI_BASE_URL"
    assert slots["flux"].env_base_url_var == "LUNA_BFL_BASE_URL"
    # provisionable signal: every keyed provider declares a base-url var
    assert all(s.env_base_url_var for s in slots.values())


def test_no_core_imports_in_source() -> None:
    for py in PKG.rglob("*.py"):
        for line in py.read_text().splitlines():
            s = line.strip()
            if s.startswith(("import luna", "from luna")) and "luna_sdk" not in s:
                raise AssertionError(f"{py.name}: forbidden core import: {s}")


# ---------------- catalog helpers ----------------
def test_resolve_model_aliases() -> None:
    assert providers.resolve_model("best").key == "nano-banana-pro"
    assert providers.resolve_model(None).key == "nano-banana-pro"
    assert providers.resolve_model("gemini").key == "nano-banana-pro"
    assert providers.resolve_model("dalle").key == "gpt-image-1"
    assert providers.resolve_model("flux").key == "flux-pro"
    assert providers.resolve_model("nope") is None


def test_clean_aspect() -> None:
    assert providers.clean_aspect("16:9") == "16:9"
    assert providers.clean_aspect("bogus") == "1:1"
    assert providers.clean_aspect(None) == "1:1"


def test_openai_size_mapping() -> None:
    assert providers._aspect_to_size("16:9") == "1536x1024"
    assert providers._aspect_to_size("9:16") == "1024x1536"
    assert providers._aspect_to_size("1:1") == "1024x1024"


# ---------------- storage ----------------
def test_storage_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LUNA_IMAGE_GEN_DIR", str(tmp_path))
    saved = storage.save_image(PNG, "image/png")
    assert saved["url"].startswith("/api/p/plugin-image-gen/file/")
    assert saved["bytes"] == len(PNG)
    name = saved["id"]
    path = storage.resolve(name)
    assert path is not None and path.read_bytes() == PNG


def test_storage_rejects_traversal(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LUNA_IMAGE_GEN_DIR", str(tmp_path))
    assert storage.resolve("../etc/passwd") is None
    assert storage.resolve("a/b.png") is None
    assert storage.resolve("missing.png") is None


# ---------------- render ----------------
def test_embed_contains_url_and_escapes() -> None:
    html = render_image_embed("/api/p/plugin-image-gen/file/x.png", prompt="<b>cat</b>", model_label="Nano Banana Pro")
    assert "/api/p/plugin-image-gen/file/x.png" in html
    assert "<img" in html
    assert "Nano Banana Pro" in html
    assert "<b>cat</b>" not in html
    assert "&lt;b&gt;" in html


# ---------------- provider logic (mocked HTTP) ----------------
@pytest.mark.asyncio
class TestProviders:
    async def test_gemini_generate_returns_bytes(self) -> None:
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["url"] = str(req.url)
            body = json.loads(req.content)
            seen["aspect"] = body["generationConfig"].get("imageConfig", {}).get("aspectRatio")
            inline = {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(PNG).decode()}}
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [inline]}}]})

        m = providers.resolve_model("nano-banana-pro")
        out = await providers.generate(m, "a cat", aspect="16:9", api_key="k", client=_client(handler))
        assert out["image_bytes"] == PNG
        assert "gemini-3-pro-image:generateContent" in seen["url"]
        assert seen["aspect"] == "16:9"

    async def test_gemini_auth_error(self) -> None:
        m = providers.resolve_model("nano-banana")
        out = await providers.generate(
            m, "x", api_key="bad", client=_client(lambda r: httpx.Response(403, text="no"))
        )
        assert out["error"] == "gemini api key missing"

    async def test_gemini_no_image_returns_error(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "refused"}]}}]})

        m = providers.resolve_model("nano-banana-pro")
        out = await providers.generate(m, "x", api_key="k", client=_client(handler))
        assert out["error"] == "no image"
        assert "refused" in out["detail"]

    async def test_openai_generate_returns_bytes(self) -> None:
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["url"] = str(req.url)
            seen["auth"] = req.headers.get("authorization")
            return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}]})

        m = providers.resolve_model("gpt-image-1")
        out = await providers.generate(m, "a dog", aspect="1:1", api_key="sk-x", client=_client(handler))
        assert out["image_bytes"] == PNG
        assert "images/generations" in seen["url"]
        assert seen["auth"] == "Bearer sk-x"

    async def test_openai_edit_returns_bytes(self) -> None:
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["url"] = str(req.url)
            return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}]})

        m = providers.resolve_model("gpt-image-1")
        out = await providers.edit(m, "make it red", (PNG, "image/png"), api_key="sk", client=_client(handler))
        assert out["image_bytes"] == PNG
        assert "images/edits" in seen["url"]

    async def test_flux_submit_poll_download(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            url = str(req.url)
            if req.method == "POST" and url.endswith("/v1/flux-pro-1.1"):
                assert req.headers.get("x-key") == "bfl"
                return httpx.Response(200, json={"id": "abc", "polling_url": "https://api.bfl.ai/v1/get_result?id=abc"})
            if "get_result" in url:
                return httpx.Response(200, json={"status": "Ready", "result": {"sample": "https://cdn.example/img.png"}})
            return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

        m = providers.resolve_model("flux-pro")
        out = await providers.generate(m, "a city", aspect="1:1", api_key="bfl", client=_client(handler))
        assert out["image_bytes"] == PNG
        assert out["mime"] == "image/png"

    async def test_flux_error_status(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            if req.method == "POST":
                return httpx.Response(200, json={"id": "x", "polling_url": "https://api.bfl.ai/v1/get_result?id=x"})
            return httpx.Response(200, json={"status": "Error", "result": None})

        m = providers.resolve_model("flux-pro")
        out = await providers.generate(m, "x", api_key="bfl", client=_client(handler), )
        assert out["error"] == "flux generation failed"

    async def test_base_url_override_routes_gemini(self) -> None:
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["url"] = str(req.url)
            inline = {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(PNG).decode()}}
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [inline]}}]})

        m = providers.resolve_model("nano-banana-pro")
        out = await providers.generate(
            m, "x", api_key="tok", client=_client(handler),
            base_url="https://gw.example/proxy/gemini",
        )
        assert out["image_bytes"] == PNG
        assert seen["url"].startswith("https://gw.example/proxy/gemini/models/")

    async def test_base_url_override_routes_openai(self) -> None:
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["url"] = str(req.url)
            return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}]})

        m = providers.resolve_model("gpt-image-1")
        await providers.generate(
            m, "x", api_key="tok", client=_client(handler),
            base_url="https://gw.example/proxy/openai",
        )
        assert seen["url"] == "https://gw.example/proxy/openai/images/generations"

    async def test_base_url_override_routes_flux(self) -> None:
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            url = str(req.url)
            if req.method == "POST":
                seen["submit"] = url
                return httpx.Response(200, json={"id": "x", "polling_url": url.replace("flux-pro-1.1", "get_result")})
            if "get_result" in url:
                return httpx.Response(200, json={"status": "Ready", "result": {"sample": "https://cdn/x.png"}})
            return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

        m = providers.resolve_model("flux-pro")
        out = await providers.generate(
            m, "x", api_key="tok", client=_client(handler),
            base_url="https://gw.example/proxy/bfl",
        )
        assert out["image_bytes"] == PNG
        assert seen["submit"] == "https://gw.example/proxy/bfl/v1/flux-pro-1.1"

    async def test_empty_prompt_rejected(self) -> None:
        m = providers.resolve_model("best")
        out = await providers.generate(m, "   ", api_key="k")
        assert out["error"] == "empty prompt"

    async def test_missing_key_rejected(self) -> None:
        m = providers.resolve_model("best")
        out = await providers.generate(m, "x", api_key="")
        assert out["error"] == "gemini api key missing"


# ---------------- fake context + tool wiring ----------------
class _FakeToolRegistry:
    def __init__(self) -> None:
        self.tools: dict = {}

    def register(self, plugin_name, tool_def, handler, **kwargs) -> None:
        self.tools[tool_def.name] = handler


@dataclass
class _StoredFile:
    ref: str
    url: str
    filename: str
    media_type: str
    size_bytes: int


class _FakeStorage:
    """Minimal StorageProvider stand-in: records every save()."""

    def __init__(self) -> None:
        self.saved: list[tuple[str, bytes, str | None]] = []

    async def save(self, data: bytes, *, filename: str, media_type: str | None = None) -> _StoredFile:
        self.saved.append((filename, data, media_type))
        return _StoredFile(
            ref=filename,
            url=f"/api/p/plugin-files/read/{filename}",
            filename=filename.split("/")[-1],
            media_type=media_type or "application/octet-stream",
            size_bytes=len(data),
        )


class _FakeContext:
    def __init__(self, storage=None) -> None:
        self.tool_registry = _FakeToolRegistry()
        self.vault = None
        self.skill_registry = None
        self.events = None
        self.storage = storage

    def get_env(self, _name: str) -> str | None:
        return None


async def _load(monkeypatch, key="k", storage=None) -> tuple[ImageGenPlugin, _FakeContext]:
    # Neutralize native-env key leakage from the dev machine.
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "BFL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    plugin = ImageGenPlugin()
    ctx = _FakeContext(storage=storage)
    await plugin.on_load(ctx)

    async def _fake_key(provider: str) -> str | None:
        return key

    monkeypatch.setattr(plugin, "_api_key", _fake_key)
    return plugin, ctx


@pytest.mark.asyncio
class TestToolHandlers:
    async def test_registers_all_three_tools(self, monkeypatch) -> None:
        _, ctx = await _load(monkeypatch)
        assert set(ctx.tool_registry.tools) == {"generate_image", "edit_image", "list_image_models"}

    async def test_generate_returns_json_string_with_embed(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("LUNA_IMAGE_GEN_DIR", str(tmp_path))

        async def fake_generate(model, prompt, **kwargs):
            return {"image_bytes": PNG, "mime": "image/png"}

        monkeypatch.setattr(providers, "generate", fake_generate)
        plugin, ctx = await _load(monkeypatch)
        raw = await ctx.tool_registry.tools["generate_image"](prompt="a cat", model="best", aspect_ratio="16:9")
        assert isinstance(raw, str)
        out = json.loads(raw)
        assert out["ok"] is True
        assert out["model"] == "nano-banana-pro"
        assert out["aspect_ratio"] == "16:9"
        assert out["image_url"].startswith("/api/p/plugin-image-gen/file/")
        assert "embed_iframe" in out and out["image_url"] in out["embed_iframe"]
        # the heavy bytes must NOT be inlined into the model-facing result
        assert "base64" not in raw and len(raw) < 4000

    async def test_generate_copies_into_files_when_storage_present(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("LUNA_IMAGE_GEN_DIR", str(tmp_path))

        async def fake_generate(model, prompt, **kwargs):
            return {"image_bytes": PNG, "mime": "image/png"}

        monkeypatch.setattr(providers, "generate", fake_generate)
        store = _FakeStorage()
        plugin, ctx = await _load(monkeypatch, storage=store)
        out = json.loads(await ctx.tool_registry.tools["generate_image"](prompt="a cat"))
        # the image was copied into Files under images/
        assert len(store.saved) == 1
        ref, data, media = store.saved[0]
        assert ref.startswith("images/") and data == PNG and media == "image/png"
        assert out["saved_to_files"] == ref
        assert "Saved to Files" in out["embed_iframe"]
        # inline render still uses the plugin's own public route, not the Files URL
        assert out["image_url"].startswith("/api/p/plugin-image-gen/file/")

    async def test_generate_without_storage_still_renders(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("LUNA_IMAGE_GEN_DIR", str(tmp_path))

        async def fake_generate(model, prompt, **kwargs):
            return {"image_bytes": PNG, "mime": "image/png"}

        monkeypatch.setattr(providers, "generate", fake_generate)
        plugin, ctx = await _load(monkeypatch)  # no storage provider
        out = json.loads(await ctx.tool_registry.tools["generate_image"](prompt="x"))
        assert out["ok"] is True
        assert "saved_to_files" not in out
        assert "Saved to Files" not in out["embed_iframe"]

    async def test_generate_relays_provider_error(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("LUNA_IMAGE_GEN_DIR", str(tmp_path))

        async def fake_generate(model, prompt, **kwargs):
            return {"error": "no image", "detail": "blocked"}

        monkeypatch.setattr(providers, "generate", fake_generate)
        plugin, ctx = await _load(monkeypatch)
        out = json.loads(await ctx.tool_registry.tools["generate_image"](prompt="x"))
        assert out["error"] == "no image"
        assert "embed_iframe" not in out

    async def test_generate_unknown_model(self, monkeypatch) -> None:
        plugin, ctx = await _load(monkeypatch)
        out = json.loads(await ctx.tool_registry.tools["generate_image"](prompt="x", model="not-real"))
        assert out["error"] == "unknown model"

    async def test_generate_missing_key(self, monkeypatch) -> None:
        plugin, ctx = await _load(monkeypatch, key=None)
        out = json.loads(await ctx.tool_registry.tools["generate_image"](prompt="x"))
        assert out["error"].endswith("api key missing")
        assert "gemini_api_key" in out["detail"]

    async def test_edit_requires_source(self, monkeypatch) -> None:
        plugin, ctx = await _load(monkeypatch)
        out = json.loads(await ctx.tool_registry.tools["edit_image"](prompt="make it night"))
        assert out["error"] == "no source image"

    async def test_edit_with_local_path(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("LUNA_IMAGE_GEN_DIR", str(tmp_path))
        src = tmp_path / "src.png"
        src.write_bytes(PNG)

        captured = {}

        async def fake_edit(model, prompt, image, **kwargs):
            captured["image"] = image
            return {"image_bytes": PNG, "mime": "image/png"}

        monkeypatch.setattr(providers, "edit", fake_edit)
        plugin, ctx = await _load(monkeypatch)
        out = json.loads(await ctx.tool_registry.tools["edit_image"](prompt="restyle", image_path=str(src)))
        assert out["ok"] is True
        assert captured["image"][0] == PNG

    async def test_list_models_reports_keys(self, monkeypatch) -> None:
        plugin, ctx = await _load(monkeypatch)  # _api_key returns "k" for all
        out = json.loads(await ctx.tool_registry.tools["list_image_models"]())
        keys = {m["model"] for m in out["models"]}
        assert {"nano-banana-pro", "gpt-image-1", "flux-pro"} <= keys
        assert all(m["key_configured"] for m in out["models"])
        assert next(m for m in out["models"] if m["default"])["model"] == "nano-banana-pro"
