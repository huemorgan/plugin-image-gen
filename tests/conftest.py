"""Test-only stub for `luna_sdk`.

`luna_sdk` is provided by the Luna runtime at load time, not installed from
PyPI. To unit-test the plugin's logic (and let the package import) without a
full Luna, we register a minimal stand-in with the few names the plugin imports.
The real contract is exercised inside Luna.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any


def _install_luna_sdk_stub() -> None:
    if "luna_sdk" in sys.modules:
        return

    mod = types.ModuleType("luna_sdk")

    @dataclass
    class ToolDef:
        name: str
        description: str = ""
        parameters: dict | None = None
        policy: str = "ask"
        risk_level: str = "low"
        timeout_seconds: int | None = None
        sensitive_args: list = field(default_factory=list)
        skill_gated: bool = False

    @dataclass
    class PluginManifest:
        name: str
        version: str
        description: str = ""
        tools: list = field(default_factory=list)
        routes_module: str | None = None
        capabilities: list = field(default_factory=list)

    @dataclass
    class CredentialSlot:
        slug: str
        credential_name: str
        owner: str
        env_key_var: str | None = None
        env_base_url_var: str | None = None

    class PluginContext:  # pragma: no cover - structural stand-in
        tool_registry: Any
        vault: Any
        events: Any
        skill_registry: Any

    class LunaPlugin:  # pragma: no cover - structural stand-in
        manifest: PluginManifest

        async def on_load(self, ctx: "PluginContext") -> None: ...

        async def on_unload(self) -> None: ...

        def credential_slots(self) -> list:
            return []

    mod.ToolDef = ToolDef
    mod.PluginManifest = PluginManifest
    mod.CredentialSlot = CredentialSlot
    mod.PluginContext = PluginContext
    mod.LunaPlugin = LunaPlugin
    sys.modules["luna_sdk"] = mod


_install_luna_sdk_stub()
