
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

METADATA_PATH = Path(__file__).resolve().parent.parent / "ml" / "ddxplus" / "release_evidences.json"
TOKEN_PATTERN = re.compile(r"^E_\d+(?:_@_(?:V_\d+|[-+]?\d+(?:\.\d+)?))?$")


@lru_cache(maxsize=1)
def load_evidence_metadata() -> Dict[str, Dict[str, Any]]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"DDXPlus evidence metadata not found: {METADATA_PATH}")
    payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("release_evidences.json must contain an object")
    return payload


@lru_cache(maxsize=1)
def get_valid_evidence_codes() -> Set[str]:
    valid: Set[str] = set()
    for code, item in load_evidence_metadata().items():
        data_type = str(item.get("data_type") or "B").upper()
        possible_values = item.get("possible-values") or []
        if data_type == "B" or not possible_values:
            valid.add(code)
        else:
            for value in possible_values:
                valid.add(f"{code}_@_{value}")
    return valid


def normalize_evidence_codes(values: Any) -> List[str]:
    """Extract, validate and de-duplicate DDXPlus evidence tokens.

    Gemini may return evidence values as plain strings, strings with a short
    explanation, or small objects such as ``{"code": "E_91"}``.  This helper
    accepts those shapes but only keeps tokens that really exist in
    ``release_evidences.json``.
    """
    if values is None:
        return []

    if isinstance(values, str):
        candidates: Iterable[Any] = [values]
    elif isinstance(values, (list, tuple, set)):
        candidates = values
    else:
        candidates = [values]

    valid = get_valid_evidence_codes()
    result: List[str] = []

    for item in candidates:
        if isinstance(item, dict):
            item = (
                item.get("code")
                or item.get("evidence_code")
                or item.get("id")
                or ""
            )

        text = str(item or "").strip().upper()
        found_tokens = re.findall(
            r"E_\d+(?:_@_(?:V_\d+|[-+]?\d+(?:\.\d+)?))?",
            text,
        )

        for token in found_tokens:
            token = token.strip().upper()
            if (
                TOKEN_PATTERN.fullmatch(token)
                and token in valid
                and token not in result
            ):
                result.append(token)

    return result


@lru_cache(maxsize=1)
def build_prompt_catalog() -> str:
    """Build a compact English catalog used by Gemini for Vietnamese mapping."""
    metadata = load_evidence_metadata()
    # Large repeated location value sets are declared once.
    groups: Dict[tuple, List[str]] = {}
    for code, item in metadata.items():
        values = tuple(item.get("possible-values") or [])
        if len(values) >= 20:
            groups.setdefault(values, []).append(code)

    shared_names: Dict[tuple, str] = {
        values: f"SHARED_VALUES_{idx}"
        for idx, values in enumerate(groups, start=1)
    }
    lines: List[str] = []
    for values, group_name in shared_names.items():
        # Obtain meanings from the first evidence using this set.
        first_code = groups[values][0]
        meanings = metadata[first_code].get("value_meaning") or {}
        items = []
        for value in values:
            detail = meanings.get(str(value), {}) if isinstance(meanings, dict) else {}
            meaning = detail.get("en") if isinstance(detail, dict) else None
            items.append(f"{value}={meaning or value}")
        lines.append(f"{group_name}: " + "; ".join(items))

    lines.append("EVIDENCES:")
    for code, item in metadata.items():
        kind = "antecedent" if item.get("is_antecedent") else "symptom"
        data_type = str(item.get("data_type") or "B").upper()
        question = str(item.get("question_en") or "").replace("\n", " ").strip()
        values = tuple(item.get("possible-values") or [])
        suffix = ""
        if values:
            if values in shared_names:
                suffix = f" | values={shared_names[values]}"
            else:
                meanings = item.get("value_meaning") or {}
                rendered = []
                for value in values:
                    detail = meanings.get(str(value), {}) if isinstance(meanings, dict) else {}
                    meaning = detail.get("en") if isinstance(detail, dict) else None
                    rendered.append(f"{value}={meaning or value}")
                suffix = " | values=" + "; ".join(rendered)
        lines.append(f"{code} | {kind} | type={data_type} | {question}{suffix}")
    return "\n".join(lines)
