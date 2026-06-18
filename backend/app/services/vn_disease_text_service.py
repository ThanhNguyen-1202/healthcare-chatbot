
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "ddxplus_vn_text_model.pkl"
DISEASE_MAPPING_PATH = Path(__file__).resolve().parent.parent / "mappings" / "ddxplus_disease_mapping.json"
_text_artifact: Optional[Dict[str, Any]] = None


def load_vn_text_model(force_reload: bool = False) -> Dict[str, Any]:
    global _text_artifact
    if _text_artifact is not None and not force_reload:
        return _text_artifact
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Vietnamese text model not found: {MODEL_PATH}. "
            "Run: python scripts/train_vietnamese_pipeline_model.py"
        )
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise RuntimeError("Invalid Vietnamese text model artifact")
    if "disease_mapping" not in artifact:
        artifact["disease_mapping"] = json.loads(
            DISEASE_MAPPING_PATH.read_text(encoding="utf-8")
        )
    _text_artifact = artifact
    return artifact


def get_vn_text_model_status() -> Dict[str, Any]:
    try:
        artifact = load_vn_text_model()
        return {
            "loaded": True,
            "path": str(MODEL_PATH),
            "metrics": artifact.get("metrics", {}),
            "scope": artifact.get("scope"),
        }
    except Exception as exc:
        return {"loaded": False, "path": str(MODEL_PATH), "error": str(exc)}


def predict_vietnamese_text_diseases(text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    clean_text = " ".join(str(text or "").strip().split())
    if len(clean_text) < 3:
        return []

    artifact = load_vn_text_model()
    pipeline = artifact["pipeline"]
    probabilities = pipeline.predict_proba([clean_text])[0]
    classes = np.asarray(pipeline.classes_)
    mapping = artifact.get("disease_mapping", {})
    limit = max(1, min(int(top_k or 5), 10))
    top_indices = np.argsort(probabilities)[-limit:][::-1]

    results: List[Dict[str, Any]] = []
    for index in top_indices:
        canonical_name = str(classes[int(index)])
        details = mapping.get(canonical_name, {})
        confidence = float(probabilities[int(index)])
        display_name = str(details.get("vi") or canonical_name)
        results.append({
            "name": display_name,
            "label": display_name,
            "canonical_name": canonical_name,
            "score": round(confidence, 6),
            "confidence": round(confidence, 6),
            "percent": round(confidence * 100, 2),
            "icd10": details.get("icd10"),
            "severity": details.get("severity"),
            "department": details.get("department"),
            "source": "vn_text_tfidf_lr",
        })
    return results
