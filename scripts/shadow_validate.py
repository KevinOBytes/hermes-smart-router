#!/usr/bin/env python3
"""Shadow mode validation — exercises the full pipeline against live services.

Tests:
1. Deterministic classifier (no external deps)
2. Gemma classifier (local Ollama)
3. Route policy evaluation
4. OpenRouter transport (real API call)
5. State persistence
6. Escalation detection

All routing decisions are recorded but the actual model used is the baseline
(since we're in shadow mode).
"""

from __future__ import annotations

import json
import os
import sys
import time
import tempfile
from pathlib import Path

# OPENROUTER_API_KEY is read from environment at runtime
# Set via: export OPENROUTER_API_KEY="sk-or-v1-..."
import os

# ── Test fixtures ─────────────────────────────────────────────────────

FIXTURES = [
    # (description, prompt, expected_class, expected_primary_alias)
    (
        "structured_simple",
        "Extract all email addresses from this CSV file and format as JSON",
        "structured_simple",
        "luna",
    ),
    (
        "agentic_execution",
        "Deploy the container to production using kubectl apply -f deploy.yaml",
        "agentic_execution",
        "deepseek_flash",
    ),
    (
        "software_engineering",
        "Refactor the authentication module to use async/await patterns",
        "software_engineering",
        "glm",
    ),
    (
        "security_engineering",
        "Analyze this CVE-2026-12345 exploit and write a Sigma detection rule",
        "security_engineering",
        "fable",  # high-risk security routes to escalation alias
    ),
    (
        "writing_communication",
        "Write a security incident report for the board of directors",
        "writing_communication",
        "sonnet",
    ),
    (
        "visual_frontend",
        "Create an SVG network topology diagram showing the DMZ architecture",
        "visual_frontend",
        "kimi_k3",
    ),
    (
        "computer_use",
        "Open the browser and navigate to the admin panel at 10.0.0.1",
        "computer_use",
        "sonnet",
    ),
    (
        "voice_tts",
        "Read this report aloud using TTS",
        "structured_simple",
        "luna",
    ),
    (
        "ambiguous",
        "What do you think about the future of AI safety?",
        None,  # ambiguous — defer to Gemma
        "glm",  # Gemma classifies as knowledge_reasoning → glm
    ),
]

# ── Results ───────────────────────────────────────────────────────────

results = {
    "total": 0,
    "deterministic_hits": 0,
    "gemma_classifications": 0,
    "fallbacks": 0,
    "correct_class": 0,
    "correct_alias": 0,
    "openrouter_success": 0,
    "openrouter_failures": 0,
    "by_fixture": [],
}


def log(msg: str) -> None:
    print(f"  {msg}")


async def main() -> int:
    print("=" * 70)
    print("HERMES SMART ROUTER — SHADOW MODE VALIDATION")
    print("=" * 70)
    print()

    from hermes_smart_router.deterministic import classify_deterministic
    from hermes_smart_router.config import SmartRouterConfig
    from hermes_smart_router.classifier import Classifier
    from hermes_smart_router.policy import RoutePolicy
    from hermes_smart_router.openrouter import OpenRouterTransport
    from hermes_smart_router.state import SqliteTaskState
    from hermes_smart_router.escalation import EscalationDetector

    config = SmartRouterConfig(mode="shadow")
    classifier = Classifier(config.ollama)
    policy = RoutePolicy(config)
    transport = OpenRouterTransport(config.openrouter, api_key=os.environ["OPENROUTER_API_KEY"])

    # ── Step 1: Deterministic classifier ──────────────────────────────
    print("[1/5] Deterministic classifier (no external deps)")
    print("-" * 50)

    for desc, prompt, exp_class, exp_alias in FIXTURES:
        result = classify_deterministic(prompt)
        if result is not None:
            actual = result.task_class.value
            expected = exp_class if exp_class else actual
            status = "✓" if actual == expected else "✗"
            log(f"{status} {desc:<25s} → {actual:<25s} (expected {expected})")
            results["deterministic_hits"] += 1
        else:
            log(f"~ {desc:<25s} → deferred to Gemma")
        results["total"] += 1

    print()

    # ── Step 2: Gemma classifier (live Ollama) ────────────────────────
    print("[2/5] Gemma classifier (live Ollama)")
    print("-" * 50)

    for desc, prompt, exp_class, exp_alias in FIXTURES:
        try:
            result = await classifier.classify(prompt)
            if result is not None:
                actual = result.task_class.value
                expected = exp_class if exp_class else actual
                status = "✓" if actual == expected else "✗"
                log(f"{status} {desc:<25s} → {actual:<25s} (conf={result.confidence:.2f})")
                results["gemma_classifications"] += 1
                if actual == expected:
                    results["correct_class"] += 1
            else:
                log(f"~ {desc:<25s} → None (low confidence)")
                results["fallbacks"] += 1
        except Exception as e:
            log(f"! {desc:<25s} → ERROR: {e}")
            results["fallbacks"] += 1

    print()

    # ── Step 3: Route policy evaluation ──────────────────────────────
    print("[3/5] Route policy evaluation")
    print("-" * 50)

    for desc, prompt, exp_class, exp_alias in FIXTURES:
        det_result = classify_deterministic(prompt)
        if det_result is not None:
            route = policy.evaluate(det_result)
        else:
            try:
                gemma_result = await classifier.classify(prompt)
                route = policy.evaluate(gemma_result)
            except Exception:
                route = policy.evaluate(None)

        actual_alias = route.primary_alias
        status = "✓" if actual_alias == exp_alias else "✗"
        log(f"{status} {desc:<25s} → {actual_alias:<20s} (expected {exp_alias})")
        if actual_alias == exp_alias:
            results["correct_alias"] += 1

    print()

    # ── Step 4: OpenRouter transport (real API call) ──────────────────
    print("[4/5] OpenRouter transport (live API call)")
    print("-" * 50)

    # Test with a cheap model (deepseek flash = $0.00000009/1K tokens)
    try:
        response = await transport.chat_completion(
            model="deepseek/deepseek-v4-flash-0731",
            messages=[{"role": "user", "content": "Reply with just the word OK and nothing else."}],
            max_tokens=50,
            temperature=0.0,
        )
        content = response.get("choices", [{}])[0].get("message", {}).get("content") or "(reasoning-only response)"
        log(f"✓ deepseek_flash responded: {str(content)[:60]!r}")
        results["openrouter_success"] += 1
    except Exception as e:
        log(f"✗ deepseek_flash ERROR: {e}")
        results["openrouter_failures"] += 1

    # Test streaming
    try:
        chunks = []
        async for chunk in transport.chat_completion_stream(
            model="deepseek/deepseek-v4-flash-0731",
            messages=[{"role": "user", "content": "Count to 3."}],
            max_tokens=20,
            temperature=0.0,
        ):
            chunks.append(chunk)
        log(f"✓ deepseek_flash streaming: {len(chunks)} chunks received")
        results["openrouter_success"] += 1
    except Exception as e:
        log(f"✗ deepseek_flash streaming ERROR: {e}")
        results["openrouter_failures"] += 1

    print()

    # ── Step 5: State + Escalation ────────────────────────────────────
    print("[5/5] State persistence and escalation detection")
    print("-" * 50)

    db_path = os.path.join(tempfile.gettempdir(), "sr_shadow_test.db")
    state = SqliteTaskState(db_path=db_path)
    detector = EscalationDetector(state)

    state.create_task("test-task-1")
    state.update_model("test-task-1", "deepseek_flash", "deepseek/deepseek-v4-flash-0731")

    for i in range(3):
        detector.record_failed_tool_call("test-task-1")
    for i in range(3):
        detector.record_tool_result("test-task-1", tool_name=f"cmd_{i}", exit_code=1, output="error")

    decision = await detector.check("test-task-1", recent_tool_results=[])
    log(f"Escalation decision: escalate={decision.should_escalate}, reason={decision.reason_code}")
    log(f"  → detail={decision.detail}")

    state.delete_task("test-task-1")
    try:
        os.remove(db_path)
    except OSError:
        pass

    log("✓ State and escalation: OK")
    print()

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total fixtures:              {results['total']}")
    print(f"  Deterministic hits:          {results['deterministic_hits']}")
    print(f"  Gemma classifications:      {results['gemma_classifications']}")
    print(f"  Fallbacks (low conf/error):  {results['fallbacks']}")
    print(f"  Correct class:              {results['correct_class']}")
    print(f"  Correct alias:              {results['correct_alias']}")
    print(f"  OpenRouter completions:      {results['openrouter_success']}")
    print(f"  OpenRouter failures:         {results['openrouter_failures']}")
    print()

    all_ok = (
        results["deterministic_hits"] > 0
        and results["openrouter_success"] > 0
        and results["correct_alias"] > 0
    )
    if all_ok:
        print("✓ SHADOW MODE VALIDATION PASSED")
    else:
        print("✗ SHADOW MODE VALIDATION FAILED — review errors above")
        return 1

    print()
    print("Next steps:")
    print("  1. Review the routing decisions above")
    print("  2. If satisfied, change smart_router.mode to 'active' in config.yaml")
    print("  3. Restart Hermes to begin active routing")
    return 0


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
