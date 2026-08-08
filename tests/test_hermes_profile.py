"""Tests for the Hermes provider-profile routing brain.

hermes_profile.py imports ``providers`` (Hermes internal) at module load, so
it can't be imported standalone outside a full Hermes install. These tests
inject a minimal ``providers`` stub so the routing core — classification,
session pinning (sliding TTL), alias resolution, fixed/active modes — is
testable in isolation without touching Hermes.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def providers_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a minimal `providers` package so hermes_profile imports."""

    class ProviderProfile:
        def __init__(self, **kwargs: Any) -> None:
            self.name = kwargs.get("name", "test")
            self.aliases = kwargs.get("aliases", ())
            self.base_url = kwargs.get("base_url", "https://openrouter.ai/api/v1")
            self.default_aux_model = kwargs.get("default_aux_model", "")
            self.fallback_models = kwargs.get("fallback_models", ())

        def fetch_models(self, **kwargs: Any) -> list[str] | None:
            return ["anthropic/claude-sonnet-5", "openai/gpt-5.6-luna"]

    base = types.ModuleType("providers.base")
    base.ProviderProfile = ProviderProfile  # type: ignore[attr-defined]

    pkg = types.ModuleType("providers")
    pkg.base = base  # type: ignore[attr-defined]
    pkg.register_provider = (lambda p: None)  # type: ignore[attr-defined]
    pkg.get_provider_profile = (lambda name: None)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "providers", pkg)
    monkeypatch.setitem(sys.modules, "providers.base", base)


def _make_profile() -> Any:
    from hermes_smart_router.hermes_profile import SmartRouterProfile

    return SmartRouterProfile(
        name="smart-router",
        aliases=("tko/smart-router",),
        base_url="https://openrouter.ai/api/v1",
        env_vars=("OPENROUTER_API_KEY",),
        default_aux_model="openai/gpt-5.6-luna",
    )


class TestFixedMode:
    def test_fixed_mode_uses_configured_alias(
        self, providers_stub: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import hermes_smart_router.hermes_profile as hp

        cfg = tmp_path / "config.yaml"
        cfg.write_text("smart_router:\n  mode: fixed\n  fixed_alias: sonnet\n")

        with monkeypatch.context() as m:
            m.setattr(hp, "_hermes_home", lambda: tmp_path)
            profile = _make_profile()
            profile._cfg._path = tmp_path / "config.yaml"
            profile._cfg._mtime = 0.0
            slug, alias = profile._route("fixed-session")
            assert alias == "sonnet"
            assert slug == "anthropic/claude-sonnet-5"


class TestSessionPinning:
    def test_pin_survives_within_ttl_and_slides(
        self, providers_stub: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import hermes_smart_router.hermes_profile as hp

        with monkeypatch.context() as m:
            m.setattr(hp, "_hermes_home", lambda: tmp_path)
            profile = _make_profile()
            cfg = tmp_path / "config.yaml"
            cfg.write_text("smart_router:\n  mode: active\n  task_ttl_seconds: 3600\n")
            profile._cfg._path = cfg
            profile._cfg._mtime = 0.0

            from hermes_smart_router.models import (
                ClassifierResult,
                RiskLevel,
                Sensitivity,
                TaskClass,
            )

            fake = ClassifierResult(
                task_class=TaskClass.SOFTWARE_ENGINEERING,
                risk=RiskLevel.MODERATE,
                sensitivity=Sensitivity.INTERNAL,
                confidence=0.95,
            )

            class FakeBert:
                def classify_to_result(self, text: str) -> ClassifierResult:
                    return fake

            calls = {"n": 0}

            class CountingBert:
                def classify_to_result(self, text: str) -> ClassifierResult:
                    calls["n"] += 1
                    return fake

            profile._last_user_text = "fix the failing pytest suite"
            m.setattr(hp, "get_classifier", lambda: CountingBert())
            slug1, alias1 = profile._route("s1")
            assert calls["n"] == 1

            # Second call reuses the pin without re-classifying.
            m.setattr(hp, "get_classifier", lambda: CountingBert())
            slug2, alias2 = profile._route("s1")
            assert (slug2, alias2) == (slug1, alias1)
            assert calls["n"] == 1  # no reclassification

            # Sliding TTL: force the pin near expiry; a hit refreshes it.
            with profile._lock:
                pin = profile._pins["s1"]
                profile._pins["s1"] = (pin[0], pin[1], time.time() - 3590)
            m.setattr(hp, "get_classifier", lambda: CountingBert())
            slug3, alias3 = profile._route("s1")
            assert (slug3, alias3) == (slug1, alias1)
            assert calls["n"] == 1  # still no reclassification → pin slid
            with profile._lock:
                assert time.time() - profile._pins["s1"][2] < 5


class TestAliasResolution:
    def test_resolve_alias_uses_config_override(
        self, providers_stub: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import hermes_smart_router.hermes_profile as hp

        with monkeypatch.context() as m:
            m.setattr(hp, "_hermes_home", lambda: tmp_path)
            profile = _make_profile()
            (tmp_path / "config.yaml").write_text(
                "smart_router:\n  aliases:\n    sonnet:\n      model_slug: org/custom\n"
            )
            profile._cfg._path = tmp_path / "config.yaml"
            profile._cfg._mtime = 0.0
            slug = profile._resolve_alias("sonnet", profile._cfg.get())
            assert slug == "org/custom"

    def test_resolve_alias_falls_back_to_default(
        self, providers_stub: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import hermes_smart_router.hermes_profile as hp

        with monkeypatch.context() as m:
            m.setattr(hp, "_hermes_home", lambda: tmp_path)
            profile = _make_profile()
            (tmp_path / "config.yaml").write_text("smart_router: {}\n")
            profile._cfg._path = tmp_path / "config.yaml"
            profile._cfg._mtime = 0.0
            slug = profile._resolve_alias("luna", profile._cfg.get())
            assert slug == "openai/gpt-5.6-luna"
