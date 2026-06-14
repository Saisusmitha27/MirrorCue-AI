from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from backend.core.config import settings
from backend.ml.features import BIAS_LABEL_KEYS, FEATURE_COLUMNS, extract_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = Path(settings.bias_ml_model_path)
if not DEFAULT_MODEL_PATH.is_absolute():
    DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / DEFAULT_MODEL_PATH
METADATA_PATH = MODEL_DIR / "bias_classifier_meta.json"

_SEVERITY_BY_LABEL: dict[str, str] = {
    "prestige_gap": "medium",
    "degree_branch_bias": "high",
    "cgpa_penalty": "low",
    "career_gap": "high",
    "tier2_location": "medium",
    "name_origin": "medium",
    "project_credibility": "medium",
    "gender_coded_language": "medium",
}

_EVIDENCE_TEMPLATES: dict[str, str] = {
    "prestige_gap": "Recruiter may deprioritize candidates from Tier-2/3 colleges despite relevant skills.",
    "degree_branch_bias": "Non-CSE/IT branch can trigger degree filters even when skill alignment is strong.",
    "cgpa_penalty": "CGPA below common screening thresholds may cause automatic filtering.",
    "career_gap": "Unexplained employment gaps are often treated as elevated risk signals.",
    "tier2_location": "Non-metro location may reduce perceived network strength and immediate availability.",
    "name_origin": "Name-origin cues can trigger unconscious regional or community assumptions.",
    "project_credibility": "Projects without metrics or company context may be dismissed as lightweight.",
    "gender_coded_language": "Soft-skill-heavy or gender-coded phrasing can reduce perceived technical fit.",
}


class BiasClassifier:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self._model = None
        self._thresholds: dict[str, float] = {key: 0.45 for key in BIAS_LABEL_KEYS}
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            return
        payload = joblib.load(self.model_path)
        if isinstance(payload, dict):
            self._model = payload.get("model")
            self._thresholds.update(payload.get("thresholds", {}))
        else:
            self._model = payload
        if METADATA_PATH.exists():
            try:
                meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
                self._thresholds.update(meta.get("thresholds", {}))
            except json.JSONDecodeError:
                pass

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def predict_proba(self, profile: dict[str, Any], *, resume_text: str = "") -> dict[str, float]:
        if not self.is_ready:
            return {key: 0.0 for key in BIAS_LABEL_KEYS}
        features = extract_features(profile, resume_text=resume_text)
        vector = np.asarray([[features[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
        estimators = getattr(self._model, "estimators_", None)
        if estimators is None:
            return {key: 0.0 for key in BIAS_LABEL_KEYS}

        scores: dict[str, float] = {}
        for idx, key in enumerate(BIAS_LABEL_KEYS):
            estimator = estimators[idx]
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(vector)[0]
                scores[key] = float(proba[-1]) if len(proba) > 1 else float(proba[0])
            else:
                scores[key] = float(estimator.predict(vector)[0])
        return scores

    def predict_flags(
        self,
        profile: dict[str, Any],
        *,
        resume_text: str = "",
        patterns: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        scores = self.predict_proba(profile, resume_text=resume_text)
        flags: list[dict[str, Any]] = []
        for key in BIAS_LABEL_KEYS:
            threshold = self._thresholds.get(key, 0.45)
            if scores.get(key, 0.0) < threshold:
                continue
            label = key
            if patterns and key in patterns:
                display = patterns[key].get("label", key.replace("_", " ").title())
            else:
                display = key.replace("_", " ").title()
            flags.append({
                "bias_type": key,
                "label": display,
                "severity": _SEVERITY_BY_LABEL.get(key, "medium"),
                "evidence": _EVIDENCE_TEMPLATES.get(key, "Potential unconscious bias signal detected."),
                "recruiter_decoded": _EVIDENCE_TEMPLATES.get(key, "Potential unconscious bias signal detected."),
                "confidence": round(scores[key], 3),
                "model": "xgboost",
            })
        return flags


@lru_cache(maxsize=1)
def get_bias_classifier() -> BiasClassifier:
    return BiasClassifier()


def save_model(model: Any, thresholds: dict[str, float], metrics: dict[str, Any]) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "thresholds": thresholds}, DEFAULT_MODEL_PATH)
    METADATA_PATH.write_text(
        json.dumps({"thresholds": thresholds, "metrics": metrics, "labels": BIAS_LABEL_KEYS}, indent=2),
        encoding="utf-8",
    )
    get_bias_classifier.cache_clear()
    return DEFAULT_MODEL_PATH


def get_ml_health_status() -> dict[str, Any]:
    """Return ML pipeline status for /health/ml."""
    classifier = get_bias_classifier()
    meta: dict[str, Any] = {}
    if METADATA_PATH.exists():
        try:
            meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    metrics = meta.get("metrics", {})
    return {
        "status": "ok" if classifier.is_ready else "degraded",
        "bias_classifier": {
            "ready": classifier.is_ready,
            "model_path": str(classifier.model_path),
            "macro_f1": metrics.get("macro_f1"),
            "train_size": metrics.get("train_size"),
            "labels": meta.get("labels", BIAS_LABEL_KEYS),
            "per_label_f1": metrics.get("per_label_f1"),
        },
        "rewrite_mapper": {
            "ready": True,
            "engine": "rule_based",
        },
        "config": {
            "use_ml_bias_classifier": settings.use_ml_bias_classifier,
        },
    }
