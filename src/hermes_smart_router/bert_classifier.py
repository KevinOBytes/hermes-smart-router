"""BERT prompt classifier — fast local intent detection.

Loads the trained distilbert + MLX head model once and caches it.
Falls back gracefully if the model isn't available.

The heavy ML dependencies (mlx, numpy, transformers, torch) are imported
lazily inside methods so importing this module never fails on machines
without them — ``get_classifier()`` simply returns None and the router
falls back to the ``luna`` alias.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from hermes_smart_router.models import (
    ClassifierResult,
    RiskLevel,
    Sensitivity,
    TaskClass,
)

logger = logging.getLogger(__name__)

# ── Model path ──────────────────────────────────────────────────────────
MODEL_DIR = os.path.expanduser("~/.smart-router-proxy/classifier-model")

# ── BERT categories → proxy TaskClass mapping ──────────────────────────
BERT_TO_TASKCLASS: dict[str, TaskClass] = {
    "coding": TaskClass.SOFTWARE_ENGINEERING,
    "cli": TaskClass.AGENTIC_EXECUTION,
    "computer_use": TaskClass.COMPUTER_USE,
    "general_knowledge": TaskClass.KNOWLEDGE_REASONING,
    "roleplay": TaskClass.WRITING_COMMUNICATION,
    "math": TaskClass.STRUCTURED_SIMPLE,
    "writing_editing": TaskClass.WRITING_COMMUNICATION,
    "data_analysis": TaskClass.STRUCTURED_SIMPLE,
    "security_threat": TaskClass.SECURITY_ENGINEERING,
    "planning": TaskClass.AGENTIC_EXECUTION,
}

CATEGORY_RISK: dict[str, RiskLevel] = {
    "coding": RiskLevel.MODERATE,
    "cli": RiskLevel.MODERATE,
    "computer_use": RiskLevel.MODERATE,
    "general_knowledge": RiskLevel.LOW,
    "roleplay": RiskLevel.LOW,
    "math": RiskLevel.LOW,
    "writing_editing": RiskLevel.LOW,
    "data_analysis": RiskLevel.LOW,
    "security_threat": RiskLevel.HIGH,
    "planning": RiskLevel.LOW,
}

CATEGORY_SENSITIVITY: dict[str, Sensitivity] = {
    "coding": Sensitivity.INTERNAL,
    "cli": Sensitivity.INTERNAL,
    "computer_use": Sensitivity.INTERNAL,
    "general_knowledge": Sensitivity.PUBLIC,
    "roleplay": Sensitivity.PUBLIC,
    "math": Sensitivity.PUBLIC,
    "writing_editing": Sensitivity.INTERNAL,
    "data_analysis": Sensitivity.INTERNAL,
    "security_threat": Sensitivity.CONFIDENTIAL,
    "planning": Sensitivity.INTERNAL,
}

CATEGORY_DESTRUCTIVE: set[str] = {"security_threat"}

CONFIDENCE_THRESHOLD = 0.45


# ── Singleton ──────────────────────────────────────────────────────────
_classifier: BertClassifier | None = None


def get_classifier() -> BertClassifier | None:
    """Return the cached classifier, or None if the model isn't available.

    Never raises: missing ML deps or a missing model both resolve to None,
    letting the router fall back to the ``luna`` alias.
    """
    global _classifier
    if _classifier is None:
        try:
            _classifier = BertClassifier()
        except Exception as exc:  # missing deps / model / load failure
            logger.warning("BERT classifier unavailable: %s", exc)
            return None
    return _classifier


class BertClassifier:
    """Wraps the trained distilbert + MLX classifier head."""

    def __init__(self, model_dir: str | Path = MODEL_DIR) -> None:
        # Lazy imports — these require the optional [classifier] extra.
        # Importing this module must not depend on them.
        try:
            import mlx.core as mx  # noqa: F401
            import mlx.nn as nn  # noqa: F401
            import numpy as np  # noqa: F401
            from transformers import AutoTokenizer  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "BERT classifier extra not installed. Install with: "
                "pip install 'hermes-smart-router[classifier]'"
            ) from exc

        self._mx = mx
        self._nn = nn
        self._np = np
        self._tokenizer_cls = AutoTokenizer
        self._head = nn.Module

        model_dir = str(model_dir)
        config_path = os.path.join(model_dir, "config.json")
        weights_path = os.path.join(model_dir, "weights.safetensors")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"BERT model not found at {model_dir}. "
                "Train or download from github.com/KevinOBytes/prompt-classifier"
            )

        with open(config_path) as f:
            self.config: dict[str, Any] = json.load(f)

        self.id2label = {int(k): v for k, v in self.config["id2label"].items()}
        self.num_classes = int(self.config["num_classes"])
        self.max_length = int(self.config.get("max_length", 64))

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.model = _build_head(self._nn, self.num_classes)
        self.model.load_weights(weights_path)
        mx.eval(self.model.parameters())

        from transformers import AutoModel

        self.hf_model = AutoModel.from_pretrained(self.config["base_model"])
        self.hf_model.eval()

    def _extract_features(self, texts: list[str]) -> Any:
        import torch

        mx = self._mx
        np = self._np
        all_features = []
        for text in texts:
            enc = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.hf_model(**enc)
                cls_embeds = outputs.last_hidden_state[:, 0, :].numpy()
            all_features.append(cls_embeds)

        features = np.concatenate(all_features, axis=0)
        return mx.array(features)

    def classify(self, text: str) -> tuple[str, float]:
        features = self._extract_features([text])
        logits = self.model(features)
        probs = self._mx.softmax(logits, axis=1)
        pred_idx = self._mx.argmax(logits, axis=1).item()
        confidence = probs[0, pred_idx].item()
        return self.id2label[pred_idx], confidence

    def classify_to_result(self, text: str) -> ClassifierResult | None:
        label, confidence = self.classify(text)

        if confidence < CONFIDENCE_THRESHOLD:
            logger.debug("BERT confidence %.2f below threshold, deferring", confidence)
            return None

        task_class = BERT_TO_TASKCLASS.get(label)
        if task_class is None:
            logger.debug("BERT label '%s' has no TaskClass mapping, deferring", label)
            return None

        destructive = label in CATEGORY_DESTRUCTIVE

        return ClassifierResult(
            task_class=task_class,
            risk=CATEGORY_RISK.get(label, RiskLevel.LOW),
            sensitivity=CATEGORY_SENSITIVITY.get(label, Sensitivity.INTERNAL),
            requires_tools=True,
            requires_vision=task_class == TaskClass.COMPUTER_USE,
            long_context=False,
            destructive_potential=destructive,
            confidence=confidence,
        )


def _build_head(nn: Any, num_classes: int) -> Any:
    """Small MLP on top of frozen distilbert features (defined lazily)."""

    class ClassifierHead(nn.Module):  # type: ignore[misc]  # nn is lazy Any
        def __init__(self, input_dim: int = 768, num_classes: int = num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )

        def __call__(self, x: Any) -> Any:
            return self.net(x)

    return ClassifierHead()
