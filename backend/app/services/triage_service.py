import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.mapping_loader import load_json_mapping


_TRIAGE_CONFIG = load_json_mapping("triage_config_from_medlineplus.json")

TRIAGE_LEVELS: Dict[str, Dict[str, Any]] = {
    "Cấp cứu ngay": {
        "priority": 1,
        "label": "Mức 1 - Cấp cứu ngay",
        "action": "Gọi 115 hoặc đến khoa cấp cứu/cơ sở y tế gần nhất ngay.",
    },
    "Rất khẩn": {
        "priority": 2,
        "label": "Mức 2 - Khẩn cấp",
        "action": "Nên được nhân viên y tế đánh giá sớm, ưu tiên trong ngày.",
    },
    "Khẩn mức vừa": {
        "priority": 3,
        "label": "Mức 3 - Khám sớm",
        "action": "Nên đi khám sớm nếu triệu chứng kéo dài, tăng lên hoặc có yếu tố nguy cơ.",
    },
    "Thông thường": {
        "priority": 4,
        "label": "Mức 4 - Theo dõi / khám thường",
        "action": "Có thể theo dõi và đặt lịch khám nếu triệu chứng không cải thiện.",
    },
    "Tự theo dõi": {
        "priority": 5,
        "label": "Mức 5 - Tự theo dõi",
        "action": "Theo dõi tại nhà, nghỉ ngơi, uống đủ nước và đi khám nếu nặng hơn.",
    },
}


_MEDLINE_LEVEL_MAP = {
    "1": "Cấp cứu ngay",
    "2": "Rất khẩn",
    "3": "Khẩn mức vừa",
    "4": "Thông thường",
    "5": "Tự theo dõi",
}
for source_key, target_level in _MEDLINE_LEVEL_MAP.items():
    source_level = (_TRIAGE_CONFIG.get("triage_levels", {}) or {}).get(source_key, {})
    if source_level.get("label"):
        TRIAGE_LEVELS[target_level]["label"] = f"Mức {source_key} - {source_level['label']}"
    if source_level.get("description"):
        TRIAGE_LEVELS[target_level]["action"] = source_level["description"]

TRIAGE_ALIASES: Dict[str, str] = {
    "cấp cứu": "Cấp cứu ngay",
    "cấp cứu ngay": "Cấp cứu ngay",
    "rất khẩn": "Rất khẩn",
    "khẩn cấp": "Rất khẩn",
    "khẩn": "Khẩn mức vừa",
    "khẩn mức vừa": "Khẩn mức vừa",
    "khám sớm": "Khẩn mức vừa",
    "thông thường": "Thông thường",
    "theo dõi": "Thông thường",
    "khám thường": "Thông thường",
    "tự theo dõi": "Tự theo dõi",
}
CRITICAL_RED_FLAG_KEYWORDS: List[str] = list(_TRIAGE_CONFIG.get("critical_red_flag_keywords", []))
URGENT_RED_FLAG_KEYWORDS: List[str] = list(_TRIAGE_CONFIG.get("urgent_red_flag_keywords", []))
HIGH_RISK_COMORBIDITIES: List[str] = [
    "tiểu đường",
    "đái tháo đường",
    "bệnh tim",
    "suy tim",
    "tăng huyết áp",
    "cao huyết áp",
    "hen",
    "copd",
    "suy thận",
    "xơ gan",
    "ung thư",
    "suy giảm miễn dịch",
]


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.lower().strip().split())


def normalize_triage_level(level: Optional[str]) -> str:
    text = normalize_text(level)
    if not text:
        return "Tự theo dõi"
    return TRIAGE_ALIASES.get(text, level if level in TRIAGE_LEVELS else "Tự theo dõi")


def get_triage_priority(level: Optional[str]) -> int:
    normalized = normalize_triage_level(level)
    return int(TRIAGE_LEVELS.get(normalized, TRIAGE_LEVELS["Tự theo dõi"])["priority"])


def get_higher_priority_level(current_level: Optional[str], new_level: Optional[str]) -> str:
    current = normalize_triage_level(current_level)
    new = normalize_triage_level(new_level)
    return new if get_triage_priority(new) < get_triage_priority(current) else current


def get_triage_label(level: Optional[str]) -> str:
    return str(TRIAGE_LEVELS[normalize_triage_level(level)]["label"])


def get_action_guidance(level: Optional[str]) -> str:
    return str(TRIAGE_LEVELS[normalize_triage_level(level)]["action"])


def _contains_any(text: str, keywords: List[str]) -> List[str]:
    found: List[str] = []
    for keyword in keywords:
        key = normalize_text(keyword)
        if key and key in text and key not in found:
            found.append(key)
    return found


def classify_red_flags(red_flags: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    red_flag_text = normalize_text(" ".join(red_flags or []))
    return (
        _contains_any(red_flag_text, CRITICAL_RED_FLAG_KEYWORDS),
        _contains_any(red_flag_text, URGENT_RED_FLAG_KEYWORDS),
    )


def parse_duration_days(duration: Any) -> Optional[int]:
    text = normalize_text(duration)
    if not text:
        return None
    if "hôm nay" in text or "hôm qua" in text:
        return 1
    if "hôm kia" in text:
        return 2
    if "vài ngày" in text or "mấy ngày" in text or "mấy hôm" in text:
        return 3
    match = re.search(r"(\d+)\s*(giờ|ngày|hôm|tuần|tháng|năm)", text)
    if not match:
        return None
    number = int(match.group(1))
    unit = match.group(2)
    if unit == "giờ":
        return 1
    if unit in {"ngày", "hôm"}:
        return number
    if unit == "tuần":
        return number * 7
    if unit == "tháng":
        return number * 30
    if unit == "năm":
        return number * 365
    return None


def _risk_factor_reasons(collected_data: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    try:
        age_value = int(collected_data.get("age")) if collected_data.get("age") is not None else None
    except (TypeError, ValueError):
        age_value = None
    if age_value is not None:
        if age_value < 12:
            reasons.append("người dùng thuộc nhóm trẻ em nên cần thận trọng hơn")
        elif age_value >= 65:
            reasons.append("người dùng từ 65 tuổi trở lên là nhóm nguy cơ cao")
    if collected_data.get("is_pregnant") is True:
        reasons.append("người dùng đang mang thai nên cần được đánh giá thận trọng")
    comorbidities = [normalize_text(item) for item in collected_data.get("comorbidities", []) or []]
    matched: List[str] = []
    for disease in comorbidities:
        for risk in HIGH_RISK_COMORBIDITIES:
            if risk in disease and risk not in matched:
                matched.append(risk)
    if matched:
        reasons.append("có bệnh nền/nguy cơ cao: " + ", ".join(matched[:4]))
    return reasons


def evaluate_triage(collected_data: Optional[Dict[str, Any]] = None, base_level: Optional[str] = None) -> Dict[str, Any]:
    data = collected_data or {}
    level = normalize_triage_level(base_level or "Tự theo dõi")
    reasons: List[str] = []

    critical_flags, urgent_flags = classify_red_flags(data.get("red_flags", []) or [])
    if critical_flags:
        level = get_higher_priority_level(level, "Cấp cứu ngay")
        reasons.append("có dấu hiệu nguy hiểm nghiêm trọng: " + ", ".join(critical_flags[:5]))
    elif urgent_flags:
        level = get_higher_priority_level(level, "Rất khẩn")
        reasons.append("có dấu hiệu cảnh báo cần khám sớm: " + ", ".join(urgent_flags[:5]))

    try:
        temp_value = float(data.get("temperature")) if data.get("temperature") is not None else None
    except (TypeError, ValueError):
        temp_value = None
    if temp_value is not None:
        if temp_value >= 40.0:
            level = get_higher_priority_level(level, "Rất khẩn")
            reasons.append(f"sốt rất cao {temp_value:g} độ")
        elif temp_value >= 39.0:
            level = get_higher_priority_level(level, "Khẩn mức vừa")
            reasons.append(f"sốt cao {temp_value:g} độ")
        elif temp_value >= 38.0:
            level = get_higher_priority_level(level, "Thông thường")
            reasons.append(f"có sốt {temp_value:g} độ")

    try:
        pain_value = int(data.get("pain_score")) if data.get("pain_score") is not None else None
    except (TypeError, ValueError):
        pain_value = None
    if pain_value is not None:
        if pain_value >= 9:
            level = get_higher_priority_level(level, "Rất khẩn")
            reasons.append(f"điểm đau rất cao {pain_value}/10")
        elif pain_value >= 7:
            level = get_higher_priority_level(level, "Khẩn mức vừa")
            reasons.append(f"điểm đau cao {pain_value}/10")
        elif pain_value >= 4:
            level = get_higher_priority_level(level, "Thông thường")
            reasons.append(f"có đau mức {pain_value}/10")

    duration_days = parse_duration_days(data.get("duration"))
    if duration_days is not None:
        if duration_days >= 14:
            level = get_higher_priority_level(level, "Khẩn mức vừa")
            reasons.append(f"triệu chứng kéo dài khoảng {duration_days} ngày")
        elif duration_days >= 7:
            level = get_higher_priority_level(level, "Thông thường")
            reasons.append("triệu chứng đã kéo dài nhiều ngày")

    risk_reasons = _risk_factor_reasons(data)
    if risk_reasons:
        level = get_higher_priority_level(level, "Khẩn mức vừa")
        reasons.extend(risk_reasons)

    if not reasons:
        reasons.append("chưa ghi nhận dấu hiệu nguy hiểm rõ ràng từ dữ liệu hiện có")

    priority = get_triage_priority(level)
    return {
        "triage_level": level,
        "triage_priority": priority,
        "triage_label": get_triage_label(level),
        "action_recommendation": get_action_guidance(level),
        "reasons": reasons,
        "department_override": "Cấp cứu" if priority == 1 else None,
    }
