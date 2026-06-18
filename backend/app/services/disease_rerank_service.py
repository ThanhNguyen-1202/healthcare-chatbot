import json
from pathlib import Path
from typing import Any, Dict, List


def load_rerank_rules() -> Dict[str, Any]:
    rules_path = Path(__file__).resolve().parent.parent / "mappings" / "disease_rerank_rules.json"
    if not rules_path.exists():
        return {"rules": []}
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.lower().strip().split())


def _get_name(item: Dict[str, Any]) -> str:
    return str(item.get("name") or item.get("label") or "").strip()


def rerank_disease_predictions(predictions: list, symptoms: list, top_k: int = 3) -> list:
    data = load_rerank_rules()
    rules = data.get("rules", [])

    symptom_set = {normalize_text(symptom) for symptom in symptoms or [] if normalize_text(symptom)}
    reranked = []

    for item in predictions or []:
        if not isinstance(item, dict):
            continue

        name = _get_name(item)
        if not name:
            continue

        score = float(item.get("score", 0.0) or 0.0)
        label_text = normalize_text(name)
        adjusted_score = score

        for rule in rules:
            match_any_symptoms = {normalize_text(s) for s in rule.get("match_any_symptoms", []) if normalize_text(s)}
            boost_keywords = [normalize_text(k) for k in rule.get("boost_keywords", []) if normalize_text(k)]
            penalty_keywords = [normalize_text(k) for k in rule.get("penalty_keywords", []) if normalize_text(k)]
            boost_score = float(rule.get("boost_score", 0.0) or 0.0)
            penalty_score = float(rule.get("penalty_score", 0.0) or 0.0)

            if symptom_set.intersection(match_any_symptoms):
                if any(keyword in label_text for keyword in boost_keywords):
                    adjusted_score += boost_score
                if any(keyword in label_text for keyword in penalty_keywords):
                    adjusted_score -= penalty_score

        adjusted_score = max(adjusted_score, 0.0)
        reranked.append({
            "name": name,
            "label": name,
            "score": adjusted_score,
        })

    reranked.sort(key=lambda x: x["score"], reverse=True)
    limit = max(1, min(int(top_k or 3), 5))
    top_items = reranked[:limit]
    total_score = sum(item["score"] for item in top_items)

    results = []
    for item in top_items:
        percent = round((item["score"] / total_score) * 100, 2) if total_score > 0 else 0.0
        results.append({
            "name": item["name"],
            "label": item["label"],
            "score": item["score"],
            "percent": percent,
        })
    return results
