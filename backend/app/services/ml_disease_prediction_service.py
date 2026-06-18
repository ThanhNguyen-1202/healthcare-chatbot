
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from app.services.ddxplus_evidence_catalog import normalize_evidence_codes

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "ddxplus_disease_model.pkl"
DISEASE_MAPPING_PATH = Path(__file__).resolve().parent.parent / "mappings" / "ddxplus_disease_mapping.json"
AGE_BIN_NAMES = (
    "0-1", "2-4", "5-12", "13-17", "18-24", "25-34",
    "35-44", "45-54", "55-64", "65-74", "75+",
)
_model_artifact: Optional[Dict[str, Any]] = None


def _age_bin_index(age: int) -> int:
    boundaries = (1, 4, 12, 17, 24, 34, 44, 54, 64, 74)
    return int(np.digitize([age], boundaries, right=True)[0])


def _normalize_gender(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip().lower()
    if text in {"m", "male", "nam", "man", "boy"}:
        return "M"
    if text in {"f", "female", "nữ", "nu", "woman", "girl"}:
        return "F"
    return None


def load_disease_model(force_reload: bool = False) -> Dict[str, Any]:
    global _model_artifact
    if _model_artifact is not None and not force_reload:
        return _model_artifact
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"DDXPlus model artifact not found: {MODEL_PATH}. "
            "Run: python scripts/train_ddxplus_structured_model.py"
        )
    artifact = joblib.load(MODEL_PATH)
    required = {"model", "label_encoder", "evidence_vectorizer"}
    missing = required.difference(artifact if isinstance(artifact, dict) else {})
    if missing:
        raise RuntimeError(f"Invalid DDXPlus model artifact; missing keys: {sorted(missing)}")

    # Backward compatibility: models trained by older versions did not embed
    # disease_mapping. Load it from the project JSON instead of forcing a retrain.
    if "disease_mapping" not in artifact:
        if not DISEASE_MAPPING_PATH.exists():
            raise RuntimeError(
                "DDXPlus model artifact has no disease_mapping and the fallback "
                f"mapping file was not found: {DISEASE_MAPPING_PATH}"
            )
        artifact["disease_mapping"] = json.loads(
            DISEASE_MAPPING_PATH.read_text(encoding="utf-8")
        )
    if not hasattr(artifact["model"], "predict_proba"):
        raise RuntimeError("DDXPlus classifier must support predict_proba().")
    _model_artifact = artifact
    return artifact


def get_model_status() -> Dict[str, Any]:
    artifact = load_disease_model()
    return {
        "loaded": True,
        "path": str(MODEL_PATH),
        "metadata": artifact.get("metadata", {}),
    }


def _build_feature_matrix(
    evidence_codes: List[str],
    age: Optional[int],
    gender: Optional[str],
):
    artifact = load_disease_model()
    vectorizer = artifact["evidence_vectorizer"]
    codes = normalize_evidence_codes(evidence_codes)
    evidence_text = "[" + ", ".join(repr(code) for code in codes) + "]"
    evidence_matrix = vectorizer.transform([evidence_text])

    extras = np.zeros((1, len(AGE_BIN_NAMES) + 4), dtype=np.float32)
    if age is not None:
        safe_age = min(max(int(age), 0), 110)
        extras[0, _age_bin_index(safe_age)] = 1.0
        extras[0, len(AGE_BIN_NAMES) + 2] = safe_age / 110.0
    sex = _normalize_gender(gender)
    if sex == "M":
        extras[0, len(AGE_BIN_NAMES)] = 1.0
    elif sex == "F":
        extras[0, len(AGE_BIN_NAMES) + 1] = 1.0
    extras[0, len(AGE_BIN_NAMES) + 3] = min(len(codes), 60) / 60.0
    return hstack([evidence_matrix, csr_matrix(extras)], format="csr"), codes


def predict_top_diseases(
    text: str = "",
    symptoms: Optional[List[str]] = None,
    top_k: int = 3,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    evidence_codes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Rank DDXPlus diseases from structured evidence codes.

    ``text`` and ``symptoms`` remain in the signature for compatibility. They
    are only inspected for explicit E_* tokens; natural-language mapping is
    performed by the existing Gemini intake extractor.
    """
    raw_codes = list(evidence_codes or [])
    if not raw_codes:
        joined = " ".join([text or ""] + [str(item) for item in symptoms or []])
        raw_codes = re.findall(r"E_\d+(?:_@_(?:V_\d+|[-+]?\d+(?:\.\d+)?))?", joined.upper())
    matrix, valid_codes = _build_feature_matrix(raw_codes, age=age, gender=gender)
    if not valid_codes:
        raise ValueError(
            "Không có mã bằng chứng DDXPlus hợp lệ. "
            "Cần trích xuất ddxplus_evidences trước khi dự đoán."
        )

    artifact = load_disease_model()
    probabilities = artifact["model"].predict_proba(matrix)[0]
    encoder = artifact["label_encoder"]
    mapping = artifact.get("disease_mapping", {})
    limit = max(1, min(int(top_k or 3), 5))
    top_indices = np.argsort(probabilities)[-limit:][::-1]

    results: List[Dict[str, Any]] = []
    for index in top_indices:
        canonical_name = str(encoder.inverse_transform([int(index)])[0])
        details = mapping.get(canonical_name, {})
        display_name = str(details.get("vi") or canonical_name)
        confidence = float(probabilities[index])
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
            "source": "ddxplus_structured_sgd",
            "evidence_count": len(valid_codes),
        })
    return results


def predict_top3_diseases(text: str) -> List[Dict[str, Any]]:
    return predict_top_diseases(text=text, top_k=3)
