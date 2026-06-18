
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.services.ml_disease_prediction_service import predict_top_diseases
from app.services.vn_disease_text_service import predict_vietnamese_text_diseases


def _canonical(item: Dict[str, Any]) -> str:
    return str(item.get("canonical_name") or item.get("name") or "").strip()


def _safe_predict_text(text: str, top_k: int) -> List[Dict[str, Any]]:
    try:
        return predict_vietnamese_text_diseases(text, top_k=top_k)
    except Exception:
        return []


def _safe_predict_structured(
    *, symptoms: Optional[List[str]], top_k: int, age: Optional[int],
    gender: Optional[str], evidence_codes: Optional[List[str]],
) -> List[Dict[str, Any]]:
    if not evidence_codes:
        return []
    try:
        return predict_top_diseases(
            symptoms=symptoms,
            top_k=top_k,
            age=age,
            gender=gender,
            evidence_codes=evidence_codes,
        )
    except Exception:
        return []


def predict_hybrid_diseases(
    text: str,
    symptoms: Optional[List[str]] = None,
    top_k: int = 3,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    evidence_codes: Optional[List[str]] = None,
    prediction_mode: str = "hybrid",
) -> List[Dict[str, Any]]:
    """Return ranked diseases using text, structured, or fused prediction.

    The Vietnamese model is intentionally dominant in hybrid mode because the
    structured model is excellent only when evidence codes are complete.  The
    free-text classifier was evaluated on a held-out 20% split and therefore
    avoids scoring on the same rows used for fitting.
    """
    mode = str(prediction_mode or "hybrid").strip().lower()
    candidate_k = max(5, min(int(top_k or 3) + 3, 10))
    text_predictions = _safe_predict_text(text, candidate_k)
    structured_predictions = _safe_predict_structured(
        symptoms=symptoms,
        top_k=candidate_k,
        age=age,
        gender=gender,
        evidence_codes=evidence_codes,
    )

    if mode == "text":
        return text_predictions[:max(1, min(int(top_k or 3), 5))]
    if mode == "structured":
        return structured_predictions[:max(1, min(int(top_k or 3), 5))]
    if not text_predictions:
        return structured_predictions[:max(1, min(int(top_k or 3), 5))]
    if not structured_predictions:
        return text_predictions[:max(1, min(int(top_k or 3), 5))]

    text_top_confidence = float(text_predictions[0].get("confidence", 0.0) or 0.0)
    if text_top_confidence >= 0.65:
        text_weight = 0.92
    elif text_top_confidence >= 0.35:
        text_weight = 0.86
    else:
        text_weight = 0.78
    structured_weight = 1.0 - text_weight

    details_by_label: Dict[str, Dict[str, Any]] = {}
    combined: Dict[str, float] = {}
    components: Dict[str, Dict[str, float]] = {}

    for item in text_predictions:
        label = _canonical(item)
        if not label:
            continue
        details_by_label[label] = dict(item)
        score = float(item.get("confidence", item.get("score", 0.0)) or 0.0)
        combined[label] = combined.get(label, 0.0) + text_weight * score
        components.setdefault(label, {})["text"] = score

    for item in structured_predictions:
        label = _canonical(item)
        if not label:
            continue
        details_by_label.setdefault(label, dict(item))
        score = float(item.get("confidence", item.get("score", 0.0)) or 0.0)
        combined[label] = combined.get(label, 0.0) + structured_weight * score
        components.setdefault(label, {})["structured"] = score

    ranked = sorted(combined.items(), key=lambda pair: pair[1], reverse=True)
    limit = max(1, min(int(top_k or 3), 5))
    selected = ranked[:limit]
    total = sum(score for _, score in selected) or 1.0

    results: List[Dict[str, Any]] = []
    for label, raw_score in selected:
        base = details_by_label[label]
        normalized_score = raw_score / total
        results.append({
            "name": base.get("name") or base.get("label") or label,
            "label": base.get("name") or base.get("label") or label,
            "canonical_name": label,
            "score": round(normalized_score, 6),
            "confidence": round(normalized_score, 6),
            "percent": round(normalized_score * 100, 2),
            "icd10": base.get("icd10"),
            "severity": base.get("severity"),
            "department": base.get("department"),
            "source": "hybrid_vn_text_ddxplus",
            "components": components.get(label, {}),
            "text_weight": text_weight,
            "structured_weight": structured_weight,
        })
    return results
