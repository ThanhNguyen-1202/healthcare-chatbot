import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.medical_disclaimer import MEDICAL_DISCLAIMER
from app.services.department_service import infer_department_from_symptoms
from app.services.triage_service import evaluate_triage, normalize_triage_level


DEFAULT_DEPARTMENT = "Nội tổng quát"


def _rules_path() -> Path:
    return Path(__file__).resolve().parent.parent / "mappings" / "triage_rules.json"


def load_rules() -> Dict[str, Any]:
    path = _rules_path()
    if not path.exists():
        return {"rules": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.lower().strip().split())



def _score_rule(rule: Dict[str, Any], symptom_set: set[str]) -> float:
    rule_symptoms = {normalize_text(s) for s in rule.get("symptoms", []) if normalize_text(s)}
    if not rule_symptoms:
        return 0.0

    overlap = rule_symptoms.intersection(symptom_set)
    if not overlap:
        return 0.0

    coverage = len(overlap) / max(len(rule_symptoms), 1)
    specificity = min(len(rule_symptoms), 5) * 0.05
    subset_bonus = 0.25 if rule_symptoms.issubset(symptom_set) else 0.0
    return coverage + specificity + subset_bonus


def _normalize_possible_diseases(raw_diseases: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    items = []
    for item in raw_diseases or []:
        name = item.get("name") or item.get("label")
        if not name:
            continue
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        items.append({"name": str(name), "score": score})

    items.sort(key=lambda x: x["score"], reverse=True)
    items = items[: max(1, min(int(top_k or 3), 5))]
    total_score = sum(item["score"] for item in items)

    results = []
    for item in items:
        percent = round((item["score"] / total_score) * 100, 2) if total_score > 0 else 0.0
        results.append({
            "name": item["name"],
            "score": item["score"],
            "percent": percent,
        })
    return results


def predict_from_symptoms(
    symptoms: List[str],
    red_flags: List[str] = None,
    temperature: float = None,
    pain_score: int = None,  # [CHANGED] Giữ điểm đau 0-10 cho triage.
    duration: str = None,
    comorbidities: List[str] = None,
    medications: List[str] = None,
    age: int = None,
    gender: str = None,
    is_pregnant: bool = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    data = load_rules()
    rules = data.get("rules", [])

    clean_symptoms = [normalize_text(s) for s in symptoms or [] if normalize_text(s)]
    symptom_set = set(clean_symptoms)

    scored_matches = []
    for rule in rules:
        score = _score_rule(rule, symptom_set)
        if score > 0:
            scored_matches.append((score, rule))

    scored_matches.sort(key=lambda item: item[0], reverse=True)

    if scored_matches:
        best_score, best_rule = scored_matches[0]
        base_level = normalize_triage_level(best_rule.get("triage_level", "Tự theo dõi"))
        department = best_rule.get("department") or infer_department_from_symptoms(clean_symptoms, red_flags)
        possible_diseases = _normalize_possible_diseases(best_rule.get("possible_diseases", []), top_k=top_k)
        matched = True
        message = f"Đã tìm thấy luật phù hợp với {round(best_score * 100, 1)} điểm khớp tương đối."
    else:
        base_level = "Tự theo dõi"
        department = infer_department_from_symptoms(clean_symptoms, red_flags)
        possible_diseases = []
        matched = False
        message = "Chưa tìm thấy luật bệnh cụ thể, dùng suy luận an toàn từ triệu chứng và yếu tố nguy cơ."

    triage_input = {
        "red_flags": red_flags or [],
        "temperature": temperature,
        "pain_score": pain_score,  # [CHANGED] Truyền điểm đau 0-10, không truyền severity.
        "duration": duration,
        "comorbidities": comorbidities or [],
        "medications": medications or [],
        "age": age,
        "gender": gender,
        "is_pregnant": is_pregnant,
    }
    triage = evaluate_triage(triage_input, base_level=base_level)

    final_department = triage.get("department_override") or department or DEFAULT_DEPARTMENT

    return {
        "matched": matched,
        "message": message,
        "possible_diseases": possible_diseases,
        "triage_level": triage["triage_level"],
        "triage_priority": triage["triage_priority"],
        "triage_label": triage["triage_label"],
        "action_recommendation": triage["action_recommendation"],
        "triage_reasons": triage["reasons"],
        "department": final_department,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }
