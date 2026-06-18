import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.mapping_loader import load_json_mapping


MAPPING_PATH = Path(__file__).resolve().parent.parent / "mappings" / "disease_to_department.json"
_DEPARTMENT_RULES = load_json_mapping("department_rules_from_medlineplus.json")
_department_mapping: Optional[Dict[str, str]] = None

DEFAULT_DEPARTMENT: str = str(_DEPARTMENT_RULES.get("default_department", "Nội Tổng Quát"))
SYMPTOM_DEPARTMENT_RULES: List[Dict[str, Any]] = list(_DEPARTMENT_RULES.get("symptom_department_rules", []))


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.lower().strip().split())


def _build_department_alias_lookup() -> Dict[str, str]:
    """Support both old alias->canonical and new canonical->[aliases] JSON shapes."""
    raw = _DEPARTMENT_RULES.get("department_aliases", {}) or {}
    lookup: Dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            canonical = str(key).strip()
            lookup[normalize_text(canonical)] = canonical
            for alias in value:
                alias_text = normalize_text(alias)
                if alias_text:
                    lookup[alias_text] = canonical
        else:
            lookup[normalize_text(key)] = str(value).strip()
    return lookup


DEPARTMENT_ALIASES: Dict[str, str] = _build_department_alias_lookup()


def normalize_department(department: Optional[str]) -> str:
    if not department:
        return DEFAULT_DEPARTMENT
    clean = str(department).strip()
    return DEPARTMENT_ALIASES.get(normalize_text(clean), clean)


def load_disease_to_department_mapping() -> Dict[str, str]:
    global _department_mapping
    if _department_mapping is None:
        if not MAPPING_PATH.exists():
            _department_mapping = {}
        else:
            with open(MAPPING_PATH, "r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            _department_mapping = payload if isinstance(payload, dict) else {}
    return _department_mapping


def _rule_keywords(rule: Dict[str, Any]) -> List[str]:
    """Read MedlinePlus-derived schema and remain compatible with old schema."""
    keywords: List[str] = []
    for field in ["keywords_vi_seed", "keywords", "keywords_en_from_medlineplus"]:
        for item in rule.get(field, []) or []:
            text = normalize_text(item)
            if text and text not in keywords:
                keywords.append(text)
    return keywords


def infer_department_from_symptoms(
    symptoms: Optional[List[str]] = None,
    red_flags: Optional[List[str]] = None,
) -> str:
    """Infer department from LLM-normalized symptoms and MedlinePlus seed rules."""
    text = normalize_text(" ".join((symptoms or []) + (red_flags or [])))
    if not text:
        return DEFAULT_DEPARTMENT

    best_department = DEFAULT_DEPARTMENT
    best_hits = 0
    for rule in SYMPTOM_DEPARTMENT_RULES:
        department = normalize_department(rule.get("department"))
        hits = sum(1 for keyword in _rule_keywords(rule) if keyword and keyword in text)
        if hits > best_hits:
            best_hits = hits
            best_department = department

    return best_department


def infer_department_from_diseases(
    diseases: list,
    symptoms: Optional[List[str]] = None,
    red_flags: Optional[List[str]] = None,
) -> str:
    mapping = load_disease_to_department_mapping()
    departments: List[str] = []

    for item in diseases or []:
        if not isinstance(item, dict):
            continue
        disease_name = item.get("name") or item.get("label")
        if not disease_name:
            continue
        department = mapping.get(disease_name)
        if department:
            departments.append(normalize_department(department))

    if departments:
        return Counter(departments).most_common(1)[0][0]

    return infer_department_from_symptoms(symptoms, red_flags)
