
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

REQUIRED_FIELDS: List[str] = ["main_symptom", "duration"]
OPTIONAL_FIELDS: List[str] = [
    "red_flags",
    "temperature",
    "pain_score",
    "symptom_location",
    "comorbidities",
    "medications",
    "age",
    "gender",
    "is_pregnant",
    "secondary_symptoms",
]


def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().strip().split())


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def unique_keep_order(values: List[str]) -> List[str]:
    results: List[str] = []
    for item in values:
        clean = normalize_text(item)
        if clean and clean not in results:
            results.append(clean)
    return results


def empty_intake() -> Dict[str, Any]:
    return {
        "age": None,
        "gender": None,
        "main_symptom": None,
        "duration": None,
        "temperature": None,
        "pain_score": None,
        "symptom_location": None,
        "secondary_symptoms": [],
        "red_flags": [],
        "comorbidities": [],
        "medications": [],
        "is_pregnant": None,
        "ddxplus_evidences": [],
        "ddxplus_negative_evidences": [],
        "ddxplus_rule_evidences": [],
        "ddxplus_evidence_matches": [],
    }


def merge_intake_values(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = empty_intake()
    merged.update(base or {})

    scalar_fields = [
        "age",
        "gender",
        "main_symptom",
        "duration",
        "temperature",
        "pain_score",
        "symptom_location",
        "is_pregnant",
    ]
    for field in scalar_fields:
        if is_missing_value(merged.get(field)) and not is_missing_value((extra or {}).get(field)):
            merged[field] = extra.get(field)

    main_symptom = normalize_text(merged.get("main_symptom"))
    for list_field in ["secondary_symptoms", "red_flags", "comorbidities", "medications"]:
        values = list(merged.get(list_field, []) or [])
        for item in (extra or {}).get(list_field, []) or []:
            clean = normalize_text(item)
            if clean and clean != main_symptom and clean not in values:
                values.append(clean)
        merged[list_field] = values

    evidence_values = list(merged.get("ddxplus_evidences", []) or [])
    for item in (extra or {}).get("ddxplus_evidences", []) or []:
        token = str(item or "").strip().upper()
        if token and token not in evidence_values:
            evidence_values.append(token)
    merged["ddxplus_evidences"] = evidence_values

    return merged


def build_intake_from_messages(messages: list, use_llm: bool = True) -> Dict[str, Any]:
    user_parts: List[str] = []
    for message in messages or []:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            user_parts.append(content.strip())

    if not user_parts:
        return empty_intake()

    try:
        from app.services.llm_extractor_service import (
            extract_intake_rule_only,
            extract_intake_with_llm,
        )

        joined_text = "\n".join(user_parts)
        extracted = (
            extract_intake_with_llm(joined_text)
            if use_llm
            else extract_intake_rule_only(joined_text)
        )
        merged = merge_intake_values(empty_intake(), extracted)
        for field in [
            "ddxplus_negative_evidences",
            "ddxplus_rule_evidences",
            "ddxplus_evidence_matches",
        ]:
            merged[field] = list(extracted.get(field, []) or [])
        return merged
    except Exception:
        logger.exception("LLM-only intake extraction failed")
        return empty_intake()
    
def get_missing_fields(intake: Dict[str, Any]) -> List[str]:
    intake = intake or {}
    missing: List[str] = []

    for field in REQUIRED_FIELDS:
        if is_missing_value(intake.get(field)):
            missing.append(field)

    return missing