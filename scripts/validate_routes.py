#!/usr/bin/env python3
"""Offline fixture-based route validation harness.

Reports per-class accuracy, confusion matrix, fallback rate, escalation rate,
and routing latency. Does not call any external APIs.

Usage:
    python scripts/validate_routes.py
    python scripts/validate_routes.py --show-text  # show fixture text
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hermes_smart_router.bert_classifier import get_classifier
from hermes_smart_router.config import SmartRouterConfig
from hermes_smart_router.models import TaskClass
from hermes_smart_router.policy import RoutePolicy

# ── Test fixtures ─────────────────────────────────────────────────────

FIXTURES: list[dict] = [
    # (text, expected_task_class, expected_primary_alias)
    {
        "text": "Extract all email addresses from this CSV file",
        "expected_class": TaskClass.STRUCTURED_SIMPLE,
        "expected_alias": "luna",
    },
    {
        "text": "Convert this YAML configuration to JSON",
        "expected_class": TaskClass.STRUCTURED_SIMPLE,
        "expected_alias": "luna",
    },
    {
        "text": "Deploy the container to production using kubectl",
        "expected_class": TaskClass.AGENTIC_EXECUTION,
        "expected_alias": "deepseek_flash",
    },
    {
        "text": "Run this bash script to install dependencies",
        "expected_class": TaskClass.AGENTIC_EXECUTION,
        "expected_alias": "deepseek_flash",
    },
    {
        "text": "Refactor the authentication module to use async/await",
        "expected_class": TaskClass.SOFTWARE_ENGINEERING,
        "expected_alias": "glm",
    },
    {
        "text": "Fix the failing test suite and update the CI pipeline",
        "expected_class": TaskClass.SOFTWARE_ENGINEERING,
        "expected_alias": "glm",
    },
    {
        "text": "Analyze this CVE-2026-12345 exploit and write a detection rule",
        "expected_class": TaskClass.SECURITY_ENGINEERING,
        "expected_alias": "sol",
    },
    {
        "text": "Reverse engineer this malware sample and identify the C2",
        "expected_class": TaskClass.SECURITY_ENGINEERING,
        "expected_alias": "sol",
    },
    {
        "text": "What do you think about the future of AI?",
        "expected_class": None,  # ambiguous, defers to BERT/fallback
        "expected_alias": "luna",  # fallback
    },
    {
        "text": "Write a security incident report for the board",
        "expected_class": TaskClass.WRITING_COMMUNICATION,
        "expected_alias": "sonnet",
    },
    {
        "text": "Draft a README for the new API with usage examples",
        "expected_class": TaskClass.WRITING_COMMUNICATION,
        "expected_alias": "sonnet",
    },
    {
        "text": "Create an SVG network topology diagram",
        "expected_class": TaskClass.VISUAL_FRONTEND,
        "expected_alias": "kimi_k3",
    },
    {
        "text": "Design a CSS layout for the landing page",
        "expected_class": TaskClass.VISUAL_FRONTEND,
        "expected_alias": "kimi_k3",
    },
    {
        "text": "Open the browser and navigate to the admin panel",
        "expected_class": TaskClass.COMPUTER_USE,
        "expected_alias": "sonnet",
    },
    {
        "text": "Summarize this JSON data into a table",
        "expected_class": TaskClass.STRUCTURED_SIMPLE,
        "expected_alias": "luna",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate route selection")
    parser.add_argument("--show-text", action="store_true", help="Show fixture text")
    args = parser.parse_args()

    config = SmartRouterConfig(mode="active")
    policy = RoutePolicy(config)

    results = {
        "total": 0,
        "correct_class": 0,
        "correct_alias": 0,
        "fallback": 0,
        "escalation": 0,
        "by_class": {},
        "confusion": {},
    }

    for fixture in FIXTURES:
        text = fixture["text"]
        expected_class = fixture["expected_class"]
        expected_alias = fixture["expected_alias"]

        if args.show_text:
            print(f"\n--- Fixture: {text[:60]}...")

        start = time.monotonic()

        # Run the BERT classifier (sole classifier in the current pipeline)
        classifier = get_classifier()
        result = classifier.classify_to_result(text) if classifier else None
        if result is not None:
            route = policy.evaluate(result)
        else:
            # BERT unavailable or below confidence threshold — fallback
            route = policy.evaluate(None)
            results["fallback"] += 1

        elapsed = (time.monotonic() - start) * 1000

        results["total"] += 1

        # Check class
        actual_class = route.task_class
        if expected_class is None:
            # Ambiguous — any class is acceptable, but check alias
            class_ok = True
        else:
            class_ok = actual_class == expected_class
            if class_ok:
                results["correct_class"] += 1

        # Check alias
        alias_ok = route.primary_alias == expected_alias
        if alias_ok:
            results["correct_alias"] += 1

        # Track by class
        class_name = actual_class.value if actual_class else "fallback"
        results["by_class"].setdefault(class_name, {"total": 0, "correct": 0})
        results["by_class"][class_name]["total"] += 1
        if class_ok:
            results["by_class"][class_name]["correct"] += 1

        # Confusion matrix
        expected_name = expected_class.value if expected_class else "ambiguous"
        results["confusion"].setdefault(expected_name, {})
        results["confusion"][expected_name].setdefault(class_name, 0)
        results["confusion"][expected_name][class_name] += 1

        # Escalation check
        if route.reason_code.value.startswith("escalation"):
            results["escalation"] += 1

        if args.show_text:
            print(f"  Route: {route.primary_alias} (expected: {expected_alias})")
            print(f"  Class: {actual_class} (expected: {expected_class})")
            print(f"  Reason: {route.reason_code.value}")
            print(f"  Latency: {elapsed:.1f}ms")

    # Report
    total = results["total"]
    print(f"\n{'='*60}")
    print(f"Route Validation Report")
    print(f"{'='*60}")
    print(f"Total fixtures:     {total}")
    print(f"Correct class:      {results['correct_class']}/{total} "
          f"({results['correct_class']/total*100:.1f}%)")
    print(f"Correct alias:      {results['correct_alias']}/{total} "
          f"({results['correct_alias']/total*100:.1f}%)")
    print(f"Classifier fallback: {results['fallback']}/{total} "
          f"({results['fallback']/total*100:.1f}%)")
    print(f"Escalation routes:  {results['escalation']}/{total} "
          f"({results['escalation']/total*100:.1f}%)")

    print(f"\nPer-class accuracy:")
    for class_name, stats in sorted(results["by_class"].items()):
        pct = stats["correct"] / stats["total"] * 100 if stats["total"] else 0
        print(f"  {class_name:30s}: {stats['correct']}/{stats['total']} ({pct:.0f}%)")

    print(f"\nConfusion matrix:")
    expected_names = sorted(results["confusion"].keys())
    print(f"{'':>20s}", end="")
    for name in expected_names:
        print(f"{name:>20s}", end="")
    print()
    for actual_name in expected_names:
        print(f"{actual_name:>20s}", end="")
        for expected_name in expected_names:
            count = results["confusion"].get(expected_name, {}).get(actual_name, 0)
            print(f"{count:>20d}", end="")
        print()

    return 0 if results["correct_alias"] >= total * 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
